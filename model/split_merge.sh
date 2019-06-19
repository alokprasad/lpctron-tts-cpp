#split -b 99M inference_model.pb small_model.
( create small_model.aa  small_model.ab small_model.ac)
cat small_model.* > inference_model.pb
