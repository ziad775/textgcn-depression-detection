import tensorflow as tf

print("\n--- System Check ---")
print(f"TensorFlow Version: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
print(f"Number of GPUs Available: {len(gpus)}")

if gpus:
    print("\n✅ SUCCESS: TensorFlow is communicating with your Nvidia GPU!")
    for gpu in gpus:
        print(f"Details: {gpu}")
else:
    print("\n❌ ERROR: TensorFlow cannot find the GPU. It will run on the CPU.")
print("--------------------\n")