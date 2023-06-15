# this function returns the area of a mask (number of pixels)
def get_area(mask):
    area = 0
    for row in mask:
        for col in row:
            if col:
                area += 1
    return area


# this function returns the index of the mask with the largest area
def get_max_area(masks):
    max_area = 0
    idx = 0
    for i in range(len(masks)):
        if(get_area(masks[i]) > max_area):
            max_area = get_area(masks[i])
            idx = i
    return idx


# this function draws the mask on the image
def overlay_mask_on_image(image, coord):
    # Ensure the mask is in 8-bit format
    image = cv2.drawContours(image, coord, -1, (0, 0, 255), 2)
    return image


# process the mask to remove the holes in the mask and return the largest region
def process_mask(mask):
    # Identify each separate region in the mask.
    labeled_mask, num_labels = ndimage.label(mask)
    
    # Count the size of each region.
    region_sizes = np.bincount(labeled_mask.flatten())
    
    # The first region (index 0) is the background, which we don't want to consider.
    region_sizes[0] = 0
    
    # Find the largest region.
    largest_region = np.argmax(region_sizes)
    
    # Create a mask that only includes the largest region.
    largest_mask = (labeled_mask == largest_region)
    
    # Fill in the holes in this region.
    filled_mask = ndimage.binary_fill_holes(largest_mask)
    
    return filled_mask



# extract the coordinates of the segment from SAM and store them in a list
def extract_segment(mask):
    binary_mask = np.array(mask) # get the segmentation of the mask and convert it to a numpy array
    binary_mask = (binary_mask * 255).astype(np.uint8) # convert the mask to a binary mask

    binary_mask = (process_mask(binary_mask) * 255).astype(np.uint8) # returns a single pixel_array with no holes in the mask


    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # find contours from the binary image
    polygon_coords = [] # stores the coordinates of the vertices of the polygon
    
    for contour in contours:
        epsilon = 0.0001 * cv2.arcLength(contour, True) # approximate contour with accuracy proportional to the contour perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True) # approximate contour with the Douglas-Peucker algorithm

        polygon_coords.append(approx) # add the coordinates of the vertices of the polygon to the list
    return polygon_coords


# this function is to encode the image for rendering by the widget
def encode_image_existing_mask(filepath, data):
    image = cv2.imread(filepath)
    segmentations = []
    for i in data:
        segmentations.append(i[1:])
    
    for segmentation in segmentations:
        points = np.array(segmentation).reshape(-1, 2) * [image.shape[1], image.shape[0]]
        points = points.astype(int)
        image = cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    byte_im = im_buf_arr.tobytes()

    encoded = base64.b64encode(byte_im).decode('utf-8')
    return "data:image/jpg;base64,"+encoded


def encode_image(filepath):
    with open(filepath, 'rb') as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), 'utf-8')

    return "data:image/jpg;base64,"+encoded

# this function is to encode the image with mask for rendering by the widget, whilst also returning the polygon coordinates of the mask and the original height and width of the image
def encode_image_mask(filepath, boxes):
    # read in the image file
    image = cv2.imread(filepath)
    h, w = image.shape[:2]
    poly_coords_list = []
    for box in boxes:
        # convert the bbox to format expected by mask_predictor
        box = np.array([
            box['x'],
            box['y'],
            box['x'] + box['width'],
            box['y'] + box['height']
        ])

        mask_predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # predict the masks
        masks, scores, logits = mask_predictor.predict(
            box = box,
            multimask_output = True
        )

        # get the index of the mask with the largest area
        idx = get_max_area(masks)
        mask = masks[idx]

        # convert the pixel array format of the mask to a polygon coordinates
        polygon_coords = extract_segment(mask)
        poly_coords_list.append(polygon_coords)
        # overlay the mask on the image
        image = overlay_mask_on_image(image, polygon_coords)

    # convert the image with mask back to bytes
    is_success, im_buf_arr = cv2.imencode(".jpg", image)
    byte_im = im_buf_arr.tobytes()

    # encode to Base64 for rendering on the web
    encoded = base64.b64encode(byte_im).decode('utf-8')
    
    return "data:image/jpg;base64,"+encoded, poly_coords_list, h, w



# converts the list of numpy array to a list a list for easier manipulation
def numpy_to_list(numpy_arr):
    coord_list = [coord[0].tolist() for array in numpy_arr for coord in array] # Convert each numpy array in the list to a regular list and extract the inner lists into coord_list
    flat_list = [] # stores the flattened list of coordinates
    for coord in coord_list: # convert the list of lists into a flat list
        flat_list.append(coord[0]) # append the x coordinate
        flat_list.append(coord[1]) # append the y coordinate
    return flat_list


# this is used to extract the frames from a video file and output into specified directory as jpg images
def extract_frames(video_path, output_dir, frame_interval=300):
    filename = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Could not open video file: {video_path}")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    if fps >= 50:
        frame_interval *= 2
    
    frame_index = 0

    while True:
        success, frame = video.read()
        if not success: 
            break

        if frame_index % frame_interval == 0:
            output_path = os.path.join(output_dir, f"{filename}_frame_{frame_index}.png")
            cv2.imwrite(output_path, frame)

        frame_index += 1

    video.release()


# this is used to split the images into training and validation sets into the dataset folder
def split_data(src_directory, out_directory, test_size=0.2):
    os.makedirs(out_directory, exist_ok=True) # create the dataset directory

    os.makedirs(os.path.join(out_directory, 'train', 'images'), exist_ok=False) # create the train images directory
    os.makedirs(os.path.join(out_directory, 'valid', 'images'), exist_ok=False) # create the valid images directory
    os.makedirs(os.path.join(out_directory, 'train', 'labels'), exist_ok=False) # create the train labels directory
    os.makedirs(os.path.join(out_directory, 'valid', 'labels'), exist_ok=False) # create the valid labels directory

    all_files = os.listdir(src_directory) # get all the files in the source directory
    train_files, valid_files = train_test_split(all_files, test_size=test_size, random_state=42) # split the files into training and validation sets

    # Move files into the train and valid directories
    for file_name in train_files:
        shutil.copy(os.path.join(src_directory, file_name), os.path.join(out_directory, 'train', 'images', file_name))
    for file_name in valid_files:
        shutil.copy(os.path.join(src_directory, file_name), os.path.join(out_directory, 'valid', 'images', file_name))


# this is used to load the images from the specified directory and output the data in the format required for YOLO training
def load_images_from_video(img_path, vid_path, ds_path, frame_interval):
    for filename in os.listdir(vid_path):
        extract_frames(os.path.join(vid_path, filename), img_path, frame_interval)
    
    split_data(img_path, ds_path) # splits into train, valid sets and moves into ds_path folder


# this is used to format the YOLO data into appropriate txt files for use in YOLO training
def output_to_txt(data_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name, content in data_dict.items():
        # remove the file extension from the image file name
        base_name = os.path.splitext(image_name)[0]
        # create the output file name by adding .txt extension
        output_file_name = base_name + '.txt'
        output_file_path = os.path.join(output_dir, output_file_name)
        
        with open(output_file_path, 'w') as f:
            for line in content:
                line_str = [str(item) for item in line]  # Convert all items to strings
                f.write(' '.join(line_str))  # Join all items in the line with ',' as separator
                f.write('\n')  # Write a new line after each line
    


# this is used to move the data from 3-dataset to dataset for use in YOLO training
def move_files(src_img, src_label, dest_img, dest_label):
    # Check if destination directories exist, if not, create them
    os.makedirs(dest_img, exist_ok=True)
    os.makedirs(dest_label, exist_ok=True)

    all_images = os.listdir(src_img)
    all_labels = os.listdir(src_label)
    
    for image in all_images:
        shutil.copy(os.path.join(src_img, image), os.path.join(dest_img, image))

    
    for label in all_labels:
        shutil.copy(os.path.join(src_label, label), os.path.join(dest_label, label))


# this is used to move the videos from 1-source to dataset/sources
def move_source_vid(src_vid, dest_vid):
    # Check if destination directories exist, if not, create them
    os.makedirs(dest_vid, exist_ok=True)

    all_videos = os.listdir(src_vid)
    
    for video in all_videos:
        shutil.copy(os.path.join(src_vid, video), os.path.join(dest_vid, video))


# clears directory
def clear_directory(directory):
    # Be careful with this function! It deletes all files and subdirectories in the specified directory
    shutil.rmtree(directory)
    os.mkdir(directory)


# this is used to create the data.yaml (necessary for YOLO training) file in the dataset folder 
def create_yaml(labels, path, output_path, train_path="train/images", val_path="valid/images", ):
    # copies labels with the last two elemens removed
    my_dict = {i: labels[i] for i in range(len(labels))}

    data = {
        'names': my_dict,
        'path': path,
        'train': train_path,
        'val': val_path
    }

    with open(output_path, 'w') as outfile: # write the data to the yaml file
        yaml.dump(data, outfile, default_flow_style=False)


# cleans the dataset
def delete_empty_labels_and_images(label_dir, image_dir):
    # Get a list of all txt files in label directory
    label_files = glob.glob(os.path.join(label_dir, '*.txt'))
    
    for label_file in label_files:
        # Check if the file is empty
        if os.stat(label_file).st_size == 0:
            # If empty, delete the label file
            os.remove(label_file)
            
            # Construct the corresponding image file path
            image_file = os.path.join(image_dir, os.path.splitext(os.path.basename(label_file))[0] + '.png')
            
            # Delete the corresponding image file, if it exists
            if os.path.exists(image_file):
                os.remove(image_file)