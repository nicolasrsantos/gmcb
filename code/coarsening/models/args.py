import argparse
import json
import yaml
import os

from datetime import datetime

def setup_parser(filename):

	with open(filename) as f:
		args = json.load(f)
		args = json.dumps(args)
		args = yaml.safe_load(args)

	descriptions = 'description'
	if 'descriptions' in args:
		descriptions = args.pop('descriptions', None)

	parser = argparse.ArgumentParser(description=descriptions)
	parser._action_groups.pop()
	parser.register("type", "bool", str2bool)
	required = parser.add_argument_group('required arguments')
	optional = parser.add_argument_group('optional arguments')

	for key, value in args.items():
		long = key
		if 'long' in value:
			long = value.pop('long', None)
		if 'type' in value:
			value['type'] = eval(value['type'])
		args = ['-%s' % key, '--%s' % long]
		kwargs = value
		if value['required']:
			required.add_argument(*args, **kwargs)
		else:
			optional.add_argument(*args, **kwargs)

	parser._action_groups.append(required)
	parser._action_groups.append(optional)

	return parser

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def update_json(options):

	if hasattr(options, 'conf') and options.conf:
		with open(options.conf) as f:
			json_dict = json.load(f)
			argparse_dict = vars(options)
			argparse_dict.update(json_dict)

def check_output(options, output_default='out'):

	if options.output_directory is None:
		options.output_directory = os.path.dirname(os.path.abspath(options.input)) + '/'
	else:
		if not os.path.exists(options.output_directory):
			os.makedirs(options.output_directory)
	if not options.output_directory.endswith('/'):
		options.output_directory += '/'
	if hasattr(options, 'input'):
		output_default, options.extension = os.path.splitext(os.path.basename(options.input))
	if options.output is None:
		options.output = options.output_directory + output_default
	else:
		options.output = options.output_directory + options.output
	if hasattr(options, 'unique_key') and options.unique_key:
		now = datetime.now()
		options.output = options.output_directory + options.output + '_' + now.strftime('%Y%m%d%H%M%S%f')
