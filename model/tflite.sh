tflite_convert \
  --output_file=inference_model.tflite \
  --graph_def_file=inference_model.pb \
  --input_arrays=inputs,input_lengths \
  --output_arrays=model/inference/add
