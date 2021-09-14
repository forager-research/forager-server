from main import TrainingJob
import threading
import os.path

def main():
    working_lock = threading.Lock()
    working_lock.acquire()
    payload = {'train_positive_paths': ['waymo/train/1506904092688646_front.jpeg', 'waymo/train/1506904093682819_front.jpeg', 'waymo/train/1506904094676800_front.jpeg', 'waymo/train/1506904095672707_front.jpeg'], 'train_negative_paths': ['waymo/train/1506904088695574_front.jpeg', 'waymo/train/1506904089697010_front.jpeg', 'waymo/train/1506904090696344_front.jpeg', 'waymo/train/1506904091693725_front.jpeg', 'waymo/train/1507239497145438_front.jpeg', 'waymo/train/1552675808778089_front.jpeg', 'waymo/train/1550004504651292_front.jpeg', 'waymo/train/1508086852953325_front.jpeg', 'waymo/train/1557962360312397_front.jpeg', 'waymo/train/1506906090680412_front.jpeg', 'waymo/train/1552660419799043_front.jpeg', 'waymo/train/1553701514387735_front.jpeg', 'waymo/train/1557546527922405_front.jpeg', 'waymo/train/1521941572115983_front.jpeg', 'waymo/train/1553552806285759_front.jpeg', 'waymo/train/1512860036529199_front.jpeg', 'waymo/train/1550192058374415_front.jpeg', 'waymo/train/1559178305737499_front.jpeg', 'waymo/train/1521998637758363_front.jpeg', 'waymo/train/1506959820627388_front.jpeg', 'waymo/train/1553904019686166_front.jpeg', 'waymo/train/1557335968649038_front.jpeg', 'waymo/train/1507253770103541_front.jpeg', 'waymo/train/1554139647204272_front.jpeg', 'waymo/train/1557335709555987_front.jpeg',], 'train_unlabeled_paths': ['waymo/train/1553206988074568_front.jpeg', 'waymo/train/1546577006594417_front.jpeg', 'waymo/train/1554306446401663_front.jpeg'], 'val_positive_paths': [], 'val_negative_paths': [], 'val_unlabeled_paths': [], 'model_kwargs': {'max_ram': 37580963840, 'aux_labels_path': 'https://storage.googleapis.com/foragerml/aux_labels/2d2b13f9-3b30-4e51-8ab9-4e8a03ba1f03/imagenet.pickle'}, 'model_id': '2d7cda19-8732-4002-9a53-0a32b92dfb66', 'model_name': 'BGSPLIT', 'notify_url': 'http://34.82.7.82:5000/bgsplit_trainer_status'}
    payload['train_positive_paths'] = \
        [os.path.join('https://storage.googleapis.com/foragerml', x)
         for x in payload['train_positive_paths']]
    payload['train_negative_paths'] = \
        [os.path.join('https://storage.googleapis.com/foragerml', x)
         for x in payload['train_negative_paths']]
    payload['train_unlabeled_paths'] = \
        [os.path.join('https://storage.googleapis.com/foragerml', x)
         for x in payload['train_unlabeled_paths']]
    payload['_lock'] = working_lock
    payload['model_kwargs']['use_cuda'] = False
    current_job = TrainingJob(**payload)
    current_job.run()

if __name__ == "__main__":
    main()
