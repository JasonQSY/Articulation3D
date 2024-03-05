import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import imageio
import random
import math
from glob import glob

import pycocotools.mask as mask_util
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures.boxes import pairwise_iou, pairwise_ioa
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import GenericMask, ColorMode
from detectron2.config import get_cfg

import planercnn.modeling  # noqa
from planercnn.data import PlaneRCNNMapper
from planercnn.data.planercnn_transforms import axis_to_angle_offset, angle_offset_to_axis
from planercnn.visualization.unit_vector_plot import get_normal_figure
from planercnn.evaluation import ArtiEvaluator
from planercnn.config import get_planercnn_cfg_defaults
from planercnn.utils.opt_utils import track_planes, optimize_planes, check_monotonic, check_axis
from planercnn.utils.arti_vis import create_instances, PlaneRCNN_Branch, draw_pred, draw_gt, get_normal_map


def main():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)

    # command line arguments
    parser = argparse.ArgumentParser(
        description="A script that run temporal optimization and generate benchmark results."
    )
    parser.add_argument("--config", required=True, help="config.yaml")
    parser.add_argument('--load-results', action='store_true')
    parser.add_argument("--input", required=True, help="pth file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="arti_test")
    parser.add_argument("--conf-threshold", default=0.7, type=float, help="confidence threshold")
    parser.add_argument("--vis-dir", default=None, type=str, help="visualization directory")
    parser.add_argument("--vis-num", default=50, type=int, help="number of visualizations")
    args = parser.parse_args()

    # setup logger
    logger = setup_logger()

    # load predictions
    with PathManager.open(args.input, 'rb') as f:
        predictions = torch.load(f)

    # collect videos
    video_ids = []
    pred_by_video = {}
    for p in predictions:
        file_path = p['file_name']
        filename = file_path.split('/')[-1]
        file_prefix = filename.replace('.png', '')
        youtube_id = file_prefix[:11]
        splits = file_prefix.split('_')
        shot_id = int(splits[-3])
        frame_id = int(splits[-2])
        frame_offset = int(splits[-1])
        video_id = '{}_{}_{}'.format(youtube_id, shot_id, frame_id)
        if video_id not in video_ids:
            video_ids.append(video_id)
        if video_id not in pred_by_video:
            pred_by_video[video_id] = {}
        pred_by_video[video_id][frame_offset] = p

    os.makedirs(args.output, exist_ok=True)
    metadata = MetadataCatalog.get(args.dataset)
    dicts = list(DatasetCatalog.get(args.dataset))
    
    # collect ground truth
    gt_by_frame = {}
    for d in dicts:
        file_path = d['file_name']
        filename = file_path.split('/')[-1]
        file_prefix = filename.replace('.png', '')
        youtube_id = file_prefix[:11]
        splits = file_prefix.split('_')
        shot_id = int(splits[-3])
        frame_id = int(splits[-2])
        frame_offset = int(splits[-1])
        fid = '{}_{}_{}_{}'.format(youtube_id, shot_id, frame_id, frame_offset)
        #if video_id not in video_ids:
        #    video_ids.append(video_id)
        #if video_id not in pred_by_video:
        #    pred_by_video[video_id] = {}
        gt_by_frame[fid] = d


    if args.vis_num >= 0:
        video_ids = random.sample(video_ids, args.vis_num)

    
    cfg = get_cfg()
    get_planercnn_cfg_defaults(cfg)
    cfg.merge_from_file(args.config)
    model = PlaneRCNN_Branch(cfg)

    shortened_class_names = {'arti_rot':'R', 'arti_tran':'T'}
    cls_name_map = [shortened_class_names[cls] for cls in metadata.thing_classes]
    
    print("dataset {}".format(args.dataset))

    # slurm split
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        array_task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID'))
        array_task_cnt = int(os.environ.get('SLURM_ARRAY_TASK_MAX')) + 1
        n = math.ceil(len(video_ids) / array_task_cnt)
        video_ids = [video_ids[i:i + n] for i in range(0, len(video_ids), n)]
        video_ids = video_ids[array_task_id]
        output_path = os.path.join(args.output, 'predictions_{:0>4}.pth'.format(array_task_id))
        print("slurm detected: idx {} total {} lines {}".format(array_task_id, array_task_cnt, len(video_ids)))


    if args.load_results:
        pred_fpaths = glob(os.path.join(args.output, 'predictions_*.pth'))
        predictions = []
        corrs = []
        corrs_opt = []
        for pred_fpath in pred_fpaths:
            print("reading " + pred_fpath)
            with PathManager.open(pred_fpath, 'rb') as f:
                data = torch.load(f)
                prediction = data['predictions']
                predictions.extend(prediction)
                corr = data['corrs']
                corrs.extend(corr)
                corr_opt = data['corrs_opt']
                corrs_opt.extend(corr_opt)

        corrs = np.array(corrs)
        corrs = corrs[np.logical_not(np.isnan(corrs))]
        corrs = np.abs(corrs)
        corrs_opt = np.array(corrs_opt)
        corrs_opt = corrs_opt[np.logical_not(np.isnan(corrs_opt))]
        corrs_opt = np.abs(corrs_opt)
        print(corrs.mean())
        print(corrs_opt.mean())

        evaluator = ArtiEvaluator(args.dataset, cfg, False, output_dir=args.output)
        evaluator.reset()
        print('[number of predictions]: {}'.format(len(predictions)))
        evaluator._predictions = predictions
        eval_results = evaluator.evaluate()
        print(eval_results)
        return


    predictions = []
    corrs = []
    corrs_opt = []
    for video_id in tqdm(video_ids):
        org_vis_list = []
        video_path = os.path.join('/home/syqian/articulation_data/step2_filtered_clips', '{}.mp4'.format(video_id))
        if not os.path.exists(video_path):
            video_path = os.path.join('/home/syqian/articulation_data/neg_clips', '{}.mp4'.format(video_id))

        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        frames = []
        preds = []
        for i, im in enumerate(reader):
            im = cv2.resize(im, (640, 480))
            frames.append(im)
            im = im[:, :, ::-1]
            pred = model.inference(im)
            pred_dict = model.process(pred)
            p_instance = create_instances(
                pred_dict['instances'], im.shape[:2], 
                pred_planes=pred_dict['pred_plane'].numpy(), 
                pred_rot_axis=pred_dict['pred_rot_axis'],
                pred_tran_axis=pred_dict['pred_tran_axis'],
            )
            preds.append(p_instance)

            # visualization without optmization
            if args.vis_dir is not None:
                vis = Visualizer(im[:, :, ::-1])
                seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

                # surface normal
                if len(p_instance.pred_boxes) == 0:
                    normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
                else:
                    normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

                # combine visualization and generate output
                combined_vis = seg_pred
                org_vis_list.append(combined_vis)

        reader.close()

        # temporal optimization
        planes = track_planes(preds)
        opt_preds = optimize_planes(preds, planes, '3dc', frames=frames)

        corr, corr_opt = check_axis(preds, opt_preds, planes['rot'], 'debug')
        #corr, corr_opt = check_monotonic(preds, opt_preds, planes['rot'], 'debug')
        corrs.extend(corr)
        corrs_opt.extend(corr_opt)

        # evaluation
        for frame_offset in pred_by_video[video_id]:
            p = pred_by_video[video_id][frame_offset]
            img_path = p['file_name']
            img = cv2.imread(p['file_name'])
            org_width = img.shape[1]
            org_height = img.shape[0]
            scale_x = 640.0 / org_width
            scale_y = 480.0 / org_height
            img = cv2.resize(img, (640, 480))
            vis = Visualizer(img[:, :, ::-1], metadata)
            gt_vis = Visualizer(img[:, :, ::-1], metadata)

            #pdb.set_trace()
            pred = opt_preds[frame_offset]

        
            # convert p_instance to pred_dict
            image_id = p['image_id']
            opt_p = {
                'image_id': image_id,
                'file_name': p['file_name'],
                'pred_depth': p['pred_depth'],
                'instances': [],
                'pred_rot_axis': pred.pred_rot_axis,
                'pred_tran_axis': pred.pred_tran_axis,
                'pred_plane': pred.pred_planes,
            }
            for i in range(pred.pred_masks.shape[0]):
                bbox = pred.pred_boxes
                bbox = BoxMode.convert(bbox.tensor.tolist()[i], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                instance = {
                    'image_id': image_id,
                    'category_id': pred.pred_classes[i],
                    'bbox': bbox,
                    'score': pred.scores[i],
                }
                opt_p['instances'].append(instance)

            predictions.append(opt_p)



        if args.vis_dir is None:
            continue

        # keyframe visualization
        for frame_offset in pred_by_video[video_id]:
            p = pred_by_video[video_id][frame_offset]
            img_path = p['file_name']
            img = cv2.imread(p['file_name'])
            org_width = img.shape[1]
            org_height = img.shape[0]
            scale_x = 640.0 / org_width
            scale_y = 480.0 / org_height
            img = cv2.resize(img, (640, 480))
            vis = Visualizer(img[:, :, ::-1], metadata)
            gt_vis = Visualizer(img[:, :, ::-1], metadata)

            # before optimization
            org_vis = org_vis_list[frame_offset]

            # after optimization
            p_instance = opt_preds[frame_offset]

            try:
                seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map, conf_threshold=args.conf_threshold)
                if len(p_instance.pred_boxes) == 0:
                    normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
                else:
                    normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

                # gt
                gt_id = '{}_{}'.format(video_id, frame_offset)
                gt_anno = gt_by_frame[gt_id]
                gt_vis = draw_gt(gt_vis, gt_anno, metadata, cls_name_map)

                # for vis in the paper
                imageio.imwrite(os.path.join(args.vis_dir, '{}_{}_raw.png'.format(video_id, frame_offset)), org_vis)
                imageio.imwrite(os.path.join(args.vis_dir, '{}_{}_gt.png'.format(video_id, frame_offset)), gt_vis)
                imageio.imwrite(os.path.join(args.vis_dir, '{}_{}_opt.png'.format(video_id, frame_offset)), seg_pred)
            
            except:
                print("error {}_{}".format(video_id, frame_offset))
                continue

        continue

        # video visualization
        writer = imageio.get_writer(os.path.join(args.vis_dir, '{}.mp4'.format(video_id)), fps=fps)
        for i, im in (enumerate(frames)):
            p_instance = opt_preds[i]
            org_vis = org_vis_list[i]

            vis = Visualizer(im)

            seg_pred = draw_pred(vis, p_instance, metadata, cls_name_map)

            # surface normal
            if len(p_instance.pred_boxes) == 0:
                normal_vis = get_normal_map(torch.tensor([[1., 0, 0]]), torch.zeros(1, 480, 640))
            else:
                normal_vis = get_normal_map(p_instance.pred_planes, p_instance.pred_masks.cpu())

            # render it
            # plane_params = p_instance.pred_planes
            # segmentations = p_instance.pred_masks
            # reduce_size = False
            # height = im.shape[0]
            # width = im.shape[1]
            #
            #pdb.set_trace()
            #meshes, uv_maps = get_single_image_mesh_arti(plane_params, segmentations, img=im, height=height, width=width, webvis=False, reduce_size=reduce_size)
            #render_img(meshes, uv_maps)
            #basename = 'arti'
            #save_obj(os.path.join('output_debug', str(i)), basename+'_pred', meshes, decimal_places=10, uv_maps=uv_maps)
            #imageio.imwrite(os.path.join('output_debug', str(i), 'input.png'), seg_pred)

            # combine visualization and generate output
            combined_vis = np.concatenate((seg_pred, normal_vis, org_vis), axis=1)
            writer.append_data(combined_vis)

    
    if os.environ.get('SLURM_ARRAY_TASK_ID') is not None:
        #PathManager.mkdirs(output_dir)
        data = {
            'predictions': predictions,
            'corrs': corrs,
            'corrs_opt': corrs_opt,
        }
        with PathManager.open(output_path, "wb") as f:
            torch.save(data, f)
    else:
        # run evaluator
        evaluator = ArtiEvaluator(args.dataset, cfg, False, output_dir=args.output)
        evaluator.reset()
        evaluator._predictions = predictions
        eval_results = evaluator.evaluate()
        print(eval_results)
            

if __name__ == "__main__":
    main()
