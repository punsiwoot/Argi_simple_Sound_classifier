from sound_function import *

#process test data
# turn_m3a_to_wav(dir="aim_y_1.m4a")
# turn_m3a_to_wav(dir="aim_y_2.m4a")
# turn_m3a_to_wav(dir="aim_y_3.m4a")
# turn_m3a_to_wav(dir="aim_n_1.m4a")
# turn_m3a_to_wav(dir="aim_n_2.m4a")

# turn_m3a_to_wav(dir="pop_y_1.m4a")
# turn_m3a_to_wav(dir="pop_y_2.m4a")

# resample_wav(input_file="aim_y_1.wav", output_file="aim_y_1.wav", new_length=350000)
# resample_wav(input_file="aim_y_2.wav", output_file="aim_y_2.wav", new_length=350000)
# resample_wav(input_file="aim_y_3.wav", output_file="aim_y_3.wav", new_length=350000)

# resample_wav(input_file="pop_y_1.wav", output_file="pop_y_1.wav", new_length=350000)
# resample_wav(input_file="pop_y_2.wav", output_file="pop_y_2.wav", new_length=350000)

# resample_wav(input_file="aim_n_1.wav", output_file="aim_n_1.wav", new_length=350000)
# resample_wav(input_file="aim_n_2.wav", output_file="aim_n_2.wav", new_length=350000)

#set up

model = get_model_sound()

result_y_1 = get_prediction(model, 'aim_y_1.wav')
print(f"label : 1  result :  {result_y_1}")
result_y_2 = get_prediction(model, 'aim_y_2.wav')
print(f"label : 1  result :  {result_y_2}")
result_y_3 = get_prediction(model, 'aim_y_3.wav')
print(f"label : 1  result :  {result_y_3}")
result_y_4 = get_prediction(model, 'pop_y_1.wav')
print(f"label : 1  result :  {result_y_3}")
result_y_5 = get_prediction(model, 'pop_y_2.wav')
print(f"label : 1  result :  {result_y_3}")

result_n_1 = get_prediction(model, 'aim_n_1.wav')
print(f"label : 0  result :  {result_n_1}")
result_n_2 = get_prediction(model, 'aim_n_2.wav')
print(f"label : 0  result :  {result_n_2}")