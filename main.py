from sound_function import *
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#set up

model = get_model_sound()

result = get_prediction(model, 'test_yes.wav')
print(f"this is  {result}")

