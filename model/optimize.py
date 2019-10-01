
#python3 /media/alok/ws/experiments/tensorflow-1.13.1/tensorflow/python/tools/optimize_for_inference.py --input=inference_model.pb --output=opt_model.pb --frozen_graph=True --input_names=inputs,input_lengths --output_names="model/inference/add"

#/home/alok/DATA-WS/tensorflow-prebuilt/tensorflow-android-static/bazel-bin/tensorflow/python/tools/optimize_for_inference \
#--input=inference_model.pb \
#--output=optimized_inception_graph.pb \
#--frozen_graph=True \
#--input_names="inputs,input_lengths" \
#--output_names="model/inference/add"

#echo "=============================="
#/home/alok/DATA-WS/tensorflow-prebuilt/tensorflow-android-static/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=inference_model.pb
#echo "=============================="
#/home/alok/DATA-WS/tensorflow-prebuilt/tensorflow-android-static/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=optimized_inception_graph.pb

/home/alok/DATA-WS/tensorflow-prebuilt/tensorflow-android-static/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
--in_graph=inference_model.pb \
--out_graph=o2.pb \
--inputs="inputs,input_lengths" \
--outputs="model/inference/add" \
--transforms='
  strip_unused_nodes
  sort_by_execution_order'
echo "=============================="
/home/alok/DATA-WS/tensorflow-prebuilt/tensorflow-android-static/bazel-bin/tensorflow/tools/graph_transforms/summarize_graph --in_graph=o2.pb
