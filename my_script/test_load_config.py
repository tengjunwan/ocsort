import yaml

with open('./my_script/ocsort_config.yaml', 'r') as file:
    config = yaml.safe_load(file)

ocsort_params = config['OCSort']

# Example: print specific parameters
print("Detection threshold:", ocsort_params['det_thresh'])
print("Use BYTE tracker:", ocsort_params['use_byte'])

yolo_params = config['YOLO']
# Example: print specific parameters
print("YOLO model path:", yolo_params['onnx_path'])
print("YOLO confidence threshold:", yolo_params['conf_threshold'])