import argparse
from training.train import train
from testing.test import test
import model.fujishima as fujishima
import model.humphrey as humphrey
import model.korzeniowski as korzeniowski
from real_time_analysis.analyse import analyse

parser = argparse.ArgumentParser(
    description='Train and test selected CNN models for chord recognition')
parser.add_argument(
    '--train',
    action='store',
    metavar='FILENAME',
    help='Execute train process for list of songs provided in text file')
parser.add_argument(
    '--test',
    action='store',
    metavar='FILENAME',
    help='Execute test process for list of songs provided in text file')
parser.add_argument(
    '--analyse',
    action='store_true',
    help='Execute realtime analysis from microphone input')
parser.add_argument(
    '--model',
    action='store',
    required=True,
    help='Name of used model',
    choices=['fujishima', 'humphrey', 'korzeniowski'])
parser.add_argument('--restore',
                    action='store',
                    metavar='FILENAME',
                    help='Restore weights of model from provided file')
parser.add_argument(
    '--plot_confusion_matrix',
    action='store_true',
    help='Mark if you want to plot confusion matrix during testing process')
args = parser.parse_args()

if args.model == 'fujishima':
    model = fujishima.create_model()
    preprocessing_properties = fujishima.properties
elif args.model == 'humphrey':
    model = humphrey.create_model()
    preprocessing_properties = humphrey.properties
elif args.model == 'korzeniowski':
    model = korzeniowski.create_model()
    preprocessing_properties = korzeniowski.properties

if args.restore is not None:
    model.load_weights(args.restore)

if args.train is not None:
    with open(args.train, 'r') as f:
        train_audio_files = f.read().splitlines()
    train(model, args.model, train_audio_files, preprocessing_properties)

if args.test is not None:
    with open(args.test, 'r') as f:
        test_audio_files = f.read().splitlines()
    test(model, test_audio_files, preprocessing_properties,
         args.plot_confusion_matrix)

if args.analyse is not None:
    analyse(model, preprocessing_properties)
