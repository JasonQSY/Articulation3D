import shutil
import os
from tqdm import tqdm
import imageio
import cv2
import numpy as np
import git
import pdb
import json
import pandas as pd
import pdb
from glob import glob
from multiprocessing import Pool
from datetime import datetime
import random

from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


def draw_bboxes(img, anno):
    LINE_THICKNESS = 3
    height = img.shape[0]
    width = img.shape[1]

    color = (0, 255, 0)

    for bbox in anno:
        label = bbox['label']
        if label == 'rotation':
            color = (0, 255, 0)
        elif label == 'translation':
            color = (255, 0, 0)
        else:
            print("error {}".format(label))
            color = (0, 0, 255)
        xmin = int(bbox['p1']['x'] * width)
        ymin = int(bbox['p1']['y'] * height)
        xmax = int(bbox['p2']['x'] * width)
        ymax = int(bbox['p2']['y'] * height)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, LINE_THICKNESS)

    return img


def save_frames(video_name):
    print(video_name)
    input_dir = '/home/syqian/articulation_data/step2_filtered_clips'
    output_dir = '/z/syqian/articulation_data/frames_v1'
    select_frame_ids = [5, 15, 25, 35, 45, 55, 65, 75, 85]
    select_frames = []
    reader = imageio.get_reader(os.path.join(input_dir, video_name))
    for frame_id, frame in enumerate(reader):
        if frame_id in select_frame_ids:
            frame_name = video_name.replace('.mp4', '_{}.png'.format(frame_id))
            if os.path.exists(os.path.join(output_dir, frame_name)):
                continue
            imageio.imwrite(os.path.join(output_dir, frame_name), frame)


class Arti_Dataset():
    def __init__(self, root, overfit_ids=None):
        """
        load train/val/test split file
        """
        self.root = root

        # overfit configurations
        self.is_overfit = overfit_ids is not None
        self.overfit_ids = overfit_ids
        if self.is_overfit:
            self.test_youtube_ids = overfit_ids
            self.val_youtube_ids = overfit_ids
            return

        test_split_file = os.path.join(self.root, 'test.txt')
        val_split_file = os.path.join(self.root, 'val.txt')

        """
        video_id = line.strip()
        youtube_id = video_id[:11]
        splits = video_id.split('_')
        shot_id = int(splits[-2])
        frame_id = int(splits[-1])
        """

        with open(test_split_file, 'r') as f:
            self.test_youtube_ids = set([shot[:11] for shot in f.read().splitlines()])
        with open(val_split_file, 'r') as f:
            self.val_youtube_ids = set([shot[:11] for shot in f.read().splitlines()])

    def sanity_check(self):
        # neg_map_f = 'benchmark/neg_map_v2.txt'
        # with open(neg_map_f, 'r') as f:
        #     pos_neg = [l.split(',') for l in f.read().splitlines()]
        # pos2neg = {}
        # neg2pos = {}
        # for pair in pos_neg:
        #     pos2neg[pair[0]] = pair[1]
        #     neg2pos[pair[1]] = pair[0]

        
        youtube_ids = {}
        for phase in ['val', 'test','train']:
            dataset = get_arti_dicts(phase, debug=True, root=self.root)
            youtube_id = set([dp['file_name'].split('/')[-1][:11] for dp in dataset])
            youtube_ids[phase] = youtube_id
            
            # assert for each positive shot, there is a negative shot
            positive_shot_ids = []
            negative_shot_ids = []

            rot_axis_count = 0
            tran_axis_count = 0
            for dp in dataset:
                shot_id = dp['file_name'].split('/')[-1].rsplit('_', 1)[0]
                positive = len(dp['annotations']) > 0
                if positive:
                    positive_shot_ids.append(shot_id)
                    if dp['annotations'][0]['rot_axis'] is not None:
                        rot_axis_count += 1
                    if dp['annotations'][0]['tran_axis'] is not None:
                        tran_axis_count += 1
                else:
                    negative_shot_ids.append(shot_id)
                
                
            print(f"""{phase} statistics: Positive {len(positive_shot_ids)}, Negative {len(negative_shot_ids)}, 
                    Tran Axis {tran_axis_count}, Rot Axis {rot_axis_count}, Total {len(dataset)}""")
            # import pdb;pdb.set_trace()
            # for p in positive_shot_ids:
            #     if not (pos2neg[p] in negative_shot_ids):
            #         pdb.set_trace()
            # for p in negative_shot_ids:
            #     if not (neg2pos[p] in positive_shot_ids):
            #         print(p)
        # assert there is youtube id shared across split
        assert(len(youtube_ids['val'].intersection(youtube_ids['train']))==0)
        assert(len(youtube_ids['train'].intersection(youtube_ids['test']))==0)
        assert(len(youtube_ids['test'].intersection(youtube_ids['val']))==0)
        print("pass sanity check")
        


    def get_phase(self, img_name):
        """
        get the train/val/test split of the image name
        """
        prefix = os.path.basename(img_name).split('.')[0]
        youtube_id = prefix[:11]

        if youtube_id in self.test_youtube_ids:
            return 'test'
        elif youtube_id in self.val_youtube_ids:
            return 'val'
        else:
            return 'train'


    def collect_negative(self, phase, negative_frame_dir = '/home/syqian/articulation_data/frames_v1_neg', start_idx = 0):
        if not os.path.exists(negative_frame_dir):
            negative_frame_dir = '/z/syqian/articulation_data/frames_v1_neg'
        assert(os.path.exists(negative_frame_dir))
        idx = start_idx
        frames = glob(os.path.join(negative_frame_dir, '*.png'))
        dataset_dicts = []
        for filename in tqdm(frames):
            if self.get_phase(filename) != phase:
                continue
            record = {}
            try:
                height, width = cv2.imread(filename).shape[:2]
            except:
                print("error {}".format(filename))
                continue
                pdb.set_trace()
                pass
            record["file_name"] = filename
            record["image_id"] = idx
            record["height"] = 480
            record["width"] = 640
            record["annotations"] = [
            ]

            idx += 1
            dataset_dicts.append(record)
        return dataset_dicts
    
    def dump(self, phase):
        """
        Cache dataset to json file.
        """
        data_root_folder = '/home/syqian/articulation_data'
        if not os.path.exists(data_root_folder):
            data_root_folder = '/z/syqian/articulation_data'

        frame_dir = f'{data_root_folder}/frames_v1'
        anno_path = f'{data_root_folder}/step3_anno.csv'

        # load rotation axis,
        # and build a map from frame_id to rot_axis
        rot_anno_path = f'{data_root_folder}/step4_rot.csv'
        rot_map = {}
        df = pd.read_csv(rot_anno_path)
        for i in range(df.shape[0]):
            # need parse the csv file
            img_name = df['original_filename'][i]
            if type(img_name) is not str:
                img_name = df['image_url'][i].split('/')[-1]
            assert type(img_name) is str
            if type(df['status'][i]) is float and np.isnan(df['status'][i]):
                continue
            anno = json.loads(df['status'][i])
            if len(anno) < 1:
                continue
            line_seg = anno[0]
            if type(line_seg) is str:
                continue

            # now we have line_seg and img_name

            # fetch rotation axis
            # OpenCV format
            # visualization:
            # p1 = (int(line_seg['p1']['x'] * width), int(line_seg['p1']['y'] * height))
            # p2 = (int(line_seg['p2']['x'] * width), int(line_seg['p2']['y'] * height))
            # cv2.line(frame, p1, p2, (255, 0, 0), 2)
            p1 = (int(line_seg['p1']['x']*640), int(line_seg['p1']['y']*480))
            p2 = (int(line_seg['p2']['x']*640), int(line_seg['p2']['y']*480))

            rot_map[img_name] = [p1[0], p1[1], p2[0], p2[1]]
            if p1[0] == p2[0] and p1[1] == p2[1]:
                print("Line segment ends coincide", img_name)
                rot_map[img_name] = None



        # load translation axis,
        # and build a map from frame_id to rot_axis
        tran_anno_path = f'{data_root_folder}/step5_trans.csv'
        tran_map = {}
        df = pd.read_csv(tran_anno_path)
        for i in range(df.shape[0]):
            # need parse the csv file
            img_name = df['original_filename'][i]
            if type(img_name) is not str:
                img_name = df['image_url'][i].split('/')[-1]
            assert type(img_name) is str
            if type(df['status'][i]) is float and np.isnan(df['status'][i]):
                continue
            anno = json.loads(df['status'][i])
            if len(anno) < 1:
                continue
            line_seg = anno[0]
            if type(line_seg) is str:
                continue

            # now we have line_seg and img_name

            # fetch rotation axis
            # OpenCV format
            # visualization:
            # p1 = (int(line_seg['p1']['x'] * width), int(line_seg['p1']['y'] * height))
            # p2 = (int(line_seg['p2']['x'] * width), int(line_seg['p2']['y'] * height))
            # cv2.line(frame, p1, p2, (255, 0, 0), 2)
            p1 = (int(line_seg['p1']['x']*640), int(line_seg['p1']['y']*480))
            p2 = (int(line_seg['p2']['x']*640), int(line_seg['p2']['y']*480))

            tran_map[img_name] = [p1[0], p1[1], p2[0], p2[1]]
            if p1[0] == p2[0] and p1[1] == p2[1]:
                print("Line segment ends coincide", img_name)
                tran_map[img_name] = None





        # COCO format bbox annotation
        dataset_dicts = []
        idx = 0
        df = pd.read_csv(anno_path)
        num_row = df.shape[0]
        neg_count = 0
        # num_row = 30
        for i in tqdm(range(num_row)):
            img_name = df['original_filename'][i]
            if type(img_name) is not str:
                img_name = df['image_url'][i].split('/')[-1]

            assert type(img_name) is str

            if self.is_overfit:
                prefix = os.path.basename(img_name).split('.')[0]
                youtube_id = prefix[:11]
                if youtube_id not in self.overfit_ids:
                    continue
            else:
                if self.get_phase(img_name) != phase:
                    continue


            #output_path = os.path.join(anno_output_dir, img_name.replace('.png', '.txt'))
            #print(df['status'][i])
            if type(df['status'][i]) is float and np.isnan(df['status'][i]):
                continue
            anno = json.loads(df['status'][i])

            if len(anno) < 1:
                neg_count += 9
                continue

            neg_count += 9-len(anno)

            bbox = anno[0]
            if type(bbox) is str:
                continue

            
            frame_ids = [5, 15, 25, 35, 45, 55, 65, 75, 85]
            
            # format: xmin, xmax, ymin, ymax
            loose_regions = [
                [0.00, 0.35, 0.00, 0.35],
                [0.32, 0.68, 0.00, 0.35],
                [0.64, 1.00, 0.00, 0.35],
                [0.00, 0.35, 0.32, 0.68],
                [0.32, 0.68, 0.32, 0.68],
                [0.64, 1.00, 0.32, 0.68],
                [0.00, 0.35, 0.64, 1.00],
                [0.32, 0.68, 0.64, 1.00],
                [0.64, 1.00, 0.64, 1.00],
            ]
            regions = [
                [0.00, 0.33, 0.00, 0.33],
                [0.33, 0.66, 0.00, 0.33],
                [0.66, 1.00, 0.00, 0.33],
                [0.00, 0.33, 0.33, 0.66],
                [0.33, 0.66, 0.33, 0.66],
                [0.66, 1.00, 0.33, 0.66],
                [0.00, 0.33, 0.66, 1.00],
                [0.33, 0.66, 0.66, 1.00],
                [0.66, 1.00, 0.66, 1.00],
            ]


            for bbox in anno:
                label = bbox['label']
                xmin, ymin, xmax, ymax = [bbox['p1']['x'], bbox['p1']['y'], bbox['p2']['x'], bbox['p2']['y']]
                for region_id, region in enumerate(regions):
                    frame_id = frame_ids[region_id]
                    r_xmin, r_xmax, r_ymin, r_ymax = loose_regions[region_id]
                    b_xmin, b_xmax, b_ymin, b_ymax = regions[region_id]
                    if xmin >= r_xmin and xmax <= r_xmax and ymin >= r_ymin and ymax <= r_ymax:
                        # find a bbox
                        img_path = os.path.join(
                            frame_dir,
                            img_name.replace('.png', '_{}.png'.format(frame_id))
                        )

                        record = {}
                        filename = img_path
                        try:
                            height, width = cv2.imread(filename).shape[:2]
                        except:
                            print("error {}".format(img_path))
                            continue
                            pdb.set_trace()
                            pass

                        bbox_list = [(xmin-b_xmin)*3, (ymin-b_ymin)*3, (xmax-b_xmin)*3, (ymax-b_ymin)*3]
                        x1 = min(max(bbox_list[0], 0.), 1.) * 640
                        y1 = min(max(bbox_list[1], 0.), 1.) * 480
                        x2 = min(max(bbox_list[2], 0.), 1.) * 640
                        y2 = min(max(bbox_list[3], 0.), 1.) * 480

                        rot_axis = None
                        if img_name in rot_map:
                            rot_axis = rot_map[img_name]
                        tran_axis = None
                        if img_name in tran_map:
                            tran_axis = tran_map[img_name]
                            
                        if label == 'rotation':
                            category_id = 0
                            if tran_axis is not None:
                                print(f"{img_name} rot bbox has trans_axis")
                                print(anno)
                                continue
                        elif label == 'translation':
                            category_id = 1
                            if rot_axis is not None:
                                print(f"{img_name} trans bbox has rot_axis")
                                print(anno)
                                continue
                        else:
                            print(f"Annotation Error {img_name}")
                            continue
                            raise "Annotation Error"

                        record["file_name"] = filename
                        record["image_id"] = idx
                        record["height"] = 480
                        record["width"] = 640
                        record["annotations"] = [
                            {
                                "bbox": [x1, y1, x2, y2],
                                "bbox_mode": 0,
                                "category_id": category_id,
                                "rot_axis": rot_axis,
                                "tran_axis": tran_axis,
                            },
                        ]

                        idx += 1
    
                        dataset_dicts.append(record)

                        # we have find the bbox, break
                        break



        print("Positive:", len(dataset_dicts))
        print("# neg in pos:", neg_count)
        negatives = self.collect_negative(phase, start_idx = idx)
        print("Negative:", len(negatives))
        dataset_dicts.extend(negatives)
        print(f"{phase}: {len(dataset_dicts)}")

        dump_file_summary = self.assign_info(phase, dataset_dicts)
        json_file = os.path.join(self.root, f'cached_set_{phase}.json')
        with open(json_file, 'w') as outfile:
            print(f"Dumping to file {json_file}...")
            json.dump(dump_file_summary, outfile)
        return dump_file_summary

    def assign_info(self, phase, dataset_dicts):
        description = f"""Arti {phase} Dataset, assume image size is 640*480, axis and box are in absolute value. 
        Both rotation and translation are annotated.
        """
        repo = git.Repo(search_parent_directories=True)
        git_hexsha = repo.head.object.hexsha
        date_created = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        rtn = {
            'info': {
                'description': description,
                'git_hexsha': git_hexsha,
                'date_created': date_created
            },
            'categories': [
                {'id': 0, 'name': 'arti_rot'},
                {'id': 1, 'name': 'arti_tran'},
            ]
        }
        dump_file_summary = {}
        dump_file_summary.update(rtn)
        dump_file_summary['data'] = dataset_dicts

        return dump_file_summary

    def convert_boxmode(self, phase):
        """
        convert boxmode
        clip box value
        """
        json_file = os.path.join(self.root, f'cached_set_{phase}.json')
        with open(json_file, 'r') as infile:
            print(f"Reading from file {json_file}...")
            dump_file_summary = json.load(infile)
        dataset_dicts = dump_file_summary['data']
        for i in range(len(dataset_dicts)):
            for j in range(len(dataset_dicts[i]['annotations'])):
                dataset_dicts[i]['annotations'][j]['bbox_mode'] = BoxMode.XYXY_REL.value
                dataset_dicts[i]['annotations'][j]['bbox'] = list(np.clip(dataset_dicts[i]['annotations'][j]['bbox'], 0, 1))
        dump_file_summary['data'] = dataset_dicts
        with open(json_file, 'w') as outfile:
            print(f"Dumping to file {json_file}...")
            json.dump(dump_file_summary, outfile)



def get_arti_dicts(phase, debug=False, root='benchmark'):
    json_file = os.path.join(os.path.join(root, f'cached_set_{phase}.json'))
    assert(os.path.exists(json_file))
    with open(json_file, 'r') as infile:
        print(f"Loading from file {json_file}...")
        dataset_dicts = json.load(infile)['data']
    count = {}
    for i in range(len(dataset_dicts)):
        for j in range(len(dataset_dicts[i]['annotations'])):
            dataset_dicts[i]['annotations'][j]['bbox_mode'] = BoxMode(dataset_dicts[i]['annotations'][j]['bbox_mode'])

    
        if debug:
            len_ann = len(dataset_dicts[i]['annotations'])
            if len_ann not in count.keys():
                count[len_ann] = 1
            else:
                count[len_ann] += 1
    if debug:
        print(count)
    return dataset_dicts



def main():
    # main data
    """
    root = 'v5_arti_RT'
    arti_dataset = Arti_Dataset(root)
    for phase in ['val', 'test', 'train']:
        arti_dataset.dump(phase)
        pass
    arti_dataset.sanity_check()
    """

    # small subset for opt debugging
    root = 'v6_arti_RT_small'
    overfit_ids = ['2M_BZvyX0Ic', '6CrBGpt0DQw', 'DZenfiAzE_o']
    arti_dataset = Arti_Dataset(root, overfit_ids=overfit_ids)
    for phase in ['val', 'test', 'train']:
        arti_dataset.dump(phase)
        pass
    arti_dataset.sanity_check()

    """
    # overfit experiment
    root = 'v4_arti_overfit'
    arti_dataset = Arti_Dataset(root, overfit_ids=['WJhGnSBJ5Cw'])
    for phase in ['val', 'test','train']:
        arti_dataset.dump(phase)
    """


    


if __name__=='__main__':
    main()
