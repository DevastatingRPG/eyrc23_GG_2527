    if torch.cuda.is_available(): 
        dev = "cuda:0" 
    else: 
        dev = "cpu" 
    device = torch.device(dev)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    image = cv2.imread(image)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    path = "EDSR_x4.pb"   
    sr.readModel(path)   
    sr.setModel("edsr",4)   
    result = sr.upsample(image)
    model = torch.load('model.tf').to(device)
    model.eval()
    image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=False),           
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(result).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    pred = torch.argmax(target_image_pred_probs, dim=1)
    
    # pred = model.predict(image)
    class_names = ['combat', 'destroyedbuilding', 'fire', 'humanitarianaid', 'militaryvehicles']
    event = class_names[pred]
    return event