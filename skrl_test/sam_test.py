from ultralytics import SAM

# Load a model
model = SAM("models/sam2_t.pt")

# Display model information (optional)
model.info()

# Run inference
result = model("sam_test.jpg", labels=[1])[0]#.masks


print(result.masks.data.shape)
# print(len(mask.xyn))