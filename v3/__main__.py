#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
from PIL import Image
import __init__ as pedrita
from datetime import datetime as dt

def parse_args(argv) -> argparse.Namespace:
	parser = argparse.ArgumentParser(prog='pedrita')
	sub = parser.add_subparsers(dest='cmd', required=True)

	p_train = sub.add_parser('train', help='train a model')
	p_train.add_argument('--epochs', '-e', type=int, default=2)
	p_train.add_argument('--image', '-i', required=True, help='image, or folder, or nested subfolders train and test, with real and fake subfolders')
	p_train.add_argument('--owarm', '-ow', type=int, default=1, help='number of epochs to train before applying Online Hard Example Mining (OHEM)')
	p_train.add_argument('--limit', '-l', type=int, default=None, help='limit the number of training samples (for quick tests)')
	p_train.add_argument('--freeze', '-z', type=int, default=2, help='number of top-level model children to freeze (negative to count from the end)')
	p_train.add_argument('--oalpha', '-a', type=float, default=0.5, help='weight for OHEM loss when applied (between 0 and 1)')

	p_test = sub.add_parser('test', help='run prediction / evaluation')
	p_test.add_argument('--image', '-i', default=None,
		help='path to test folder (subfolders per label) to compute accuracy')
	p_test.add_argument('--cpu', action='store_true', default=False, help='force CPU usage')
	p_test.add_argument('--limit', '-l', type=int, default=None, help='limit the number of evaluation samples (for quick tests)')

	p_video = sub.add_parser('video', help='run frame-wise prediction on a video file')
	p_video.add_argument('--video', '-v', required=True, help='path to video file')
	p_video.add_argument('--nframes', '-n', type=int, default=30, help='randomly sample up to N frames from the video for prediction')
	return parser.parse_args(argv)

def main():
	argv = list(sys.argv[1:])

	model_name = Path(argv.pop(1))
	if not model_name.is_file():
		raise ValueError(f'Please provide a valid model') from None
	
	argv = parse_args(argv)
	argv.image = Path(argv.image) if argv.image else None
	pedrita.best_device(getattr(argv, 'cpu', False))
	pedrita.set_model(model_name)
	
	now = dt.now()
	dict(
		train=cli_train, 
	  	test=cli_test, 
	  	video=cli_video,
	)[argv.cmd](argv)
	print(dt.now() - now)

def cli_test(argv):
	# If evaluation requested, run and exit
	if argv.image.is_dir():
		pedrita.evaluate_folder(argv.image, limit=argv.limit)
		return
	elif not argv.image.is_file():
		raise ValueError('Please provide a valid image file') from None

	img = Image.open(argv.image)
	prob, cam_img = pedrita.heatmap(img)
	print(f'Proba real: {prob*100:.2f} %')

	if cam_img is not None:
		fname = 'heatmap.jpg'
		cam_img.save(fname)
		print(fname)

def cli_train(argv):
	if not argv.image.is_dir():
		raise ValueError('Please provide a folder') from None
	
	pedrita.train(argv.image, 
		epochs=argv.epochs, 
		limit=argv.limit, 
		freeze=argv.freeze, 
		ohwarmup=argv.owarm,
		ohalpha=argv.oalpha,
	)
	test_dir = argv.image / 'test'
	if not test_dir.is_dir(): return

	print('# Evaluate')
	pedrita.evaluate_folder(test_dir, 
		limit=argv.limit*0.2 if argv.limit else None)

def cli_video(argv):
	results = pedrita.predict_video(
		video_path=argv.video,
		num_frames=argv.nframes,
	)
	print(results)

if __name__ == '__main__': main()
