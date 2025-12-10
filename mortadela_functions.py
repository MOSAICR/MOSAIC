import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont


def get_template_pixel_size(file_path):
    template_pixel_size = 0
    # Read the info file
    with open(file_path, 'r', encoding='latin-1') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("Pixel size:"):
                template_pixel_size = line.split(":")[1].strip()
                template_pixel_size = float(template_pixel_size.split("x")[0].strip())
                break
    return template_pixel_size


# Create a function to overlay images
def overlay_images(template, maldi, max_loc, alpha=0.7):
    for c in range(0, 3):  # Loop over the color channels
        template[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] = \
            (1 - alpha) * template[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] + \
            alpha * maldi[:maldi.shape[0], :maldi.shape[1]] / 255.0 * 255  # Scale maldi to match color channels
    return template

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def rotate_image_2(image, angle):
    # Get image dimensions
    (h, w) = image.shape[:2]
    image_center = (w / 2, h / 2)
    
    # Get the rotation matrix
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    # Calculate the sine and cosine of the rotation angle
    abs_cos = abs(rot_mat[0, 0])
    abs_sin = abs(rot_mat[0, 1])
    
    # Calculate new bounding dimensions
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to account for translation
    rot_mat[0, 2] += (new_w / 2) - image_center[0]
    rot_mat[1, 2] += (new_h / 2) - image_center[1]
    
    # Perform the rotation with the new dimensions
    result = cv2.warpAffine(image, rot_mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 0, 255))  # Transparent (RGBA))
    return result


def calculate_coordinates_if_maldi_topleft_is_00(maldi_tl, template, angle):
    """
    Assume maldi_tl is (0,0) and calculate the template points accordingly. The whole system is rotated by angle. Maldi is a subimage inside template

    Args:
        maldi_tl: Top-left corner of MALDI (0,0)
        template: Template image (relative coordinates)
        angle: Rotation angle in degrees (to be applied to the whole system)
    Returns:
        Rotated coordinates of the four corners
    """

    maldi_top_left = (0, 0)


    template_top_left = (0, 0)
    template_top_right = (template.shape[1], 0)
    template_bottom_left = (0, template.shape[0])
    template_bottom_right = (template.shape[1], template.shape[0])

    ox, oy = template.shape[1] // 2, template.shape[0] // 2

    maldi_tl_rot = rotate_point(maldi_tl[0], maldi_tl[1], ox, oy, angle)
    # maldi_tr_rot = rotate_point(maldi_top_right[0], maldi_top_right[1], ox, oy, angle) --- IGNORE ---
    # maldi_bl_rot = rotate_point(maldi_bottom_left[0], maldi_bottom_left[1], ox, oy, angle) --- IGNORE ---
    # maldi_br_rot = rotate_point(maldi_bottom_right[0], maldi_bottom_right[1], ox, oy, angle) --- IGNORE ---

    template_tl_rot = rotate_point(template_top_left[0], template_top_left[1], ox, oy, angle)
    template_tr_rot = rotate_point(template_top_right[0], template_top_right[1], ox, oy, angle)
    template_bl_rot = rotate_point(template_bottom_left[0], template_bottom_left[1], ox, oy, angle)
    template_br_rot = rotate_point(template_bottom_right[0], template_bottom_right[1], ox, oy, angle)

    return template_tl_rot, template_tr_rot, template_bl_rot, template_br_rot, maldi_tl_rot


def calculate_coordinates_if_maldi_topleft_is_00_final(big_template, template, angle, biggest_add):
    """
    Assume maldi_tl is (0,0) and calculate the template points accordingly. The whole system is rotated by angle. Maldi is a subimage inside template

    Args:
        maldi_tl: Top-left corner of MALDI (0,0)
        template: Template image (relative coordinates)
        angle: Rotation angle in degrees (to be applied to the whole system)
    Returns:
        Rotated coordinates of the four corners
    """
    # plt.imshow(cv2.cvtColor(big_template, cv2.COLOR_BGR2RGB), aspect='equal')
    # plt.axis('off')

    # plt.plot(0, 0, 'ro')  #maldi top left
    # plt.plot(big_template.shape[1], 0, 'ro')  #maldi bottom left
    # plt.plot(0, big_template.shape[0], 'ro')  #maldi top right
    # plt.plot(big_template.shape[1], big_template.shape[0], 'ro')  #maldi bottom right
    
    template_top_left = (0, 0)
    template_top_right = (template.shape[1], 0)
    template_bottom_left = (0, template.shape[0])
    template_bottom_right = (template.shape[1], template.shape[0])
    # plt.plot(template_top_left[0] + biggest_add//2, template_top_left[1] + biggest_add//2, 'go')  #template top left
    # plt.plot(template_top_right[0] + biggest_add//2, template_top_right[1] + biggest_add//2, 'go')  #template top right
    # plt.plot(template_bottom_left[0] + biggest_add//2, template_bottom_left[1] + biggest_add//2, 'go')  #template bottom left
    # plt.plot(template_bottom_right[0] + biggest_add//2, template_bottom_right[1] + biggest_add//2, 'go')  #template bottom right
    # plt.plot(biggest_add//2, biggest_add//2, 'go')  #template top left
    # plt.plot(big_template.shape[1] - biggest_add//2, 0 + biggest_add//2, 'go')  #template top right
    # plt.plot(0 + biggest_add//2, big_template.shape[0] - biggest_add//2, 'go')  #template bottom left
    # plt.plot(big_template.shape[1] - biggest_add//2, big_template.shape[0] - biggest_add//2, 'go')  #template bottom right


    # plt.show()

    # plt.close()
    # print(f"big_template.shape: {big_template.shape}")
    # print(f"template.shape: {template.shape}")

    ox, oy = big_template.shape[1] // 2, big_template.shape[0] // 2


    template_tl_rot = rotate_point(biggest_add//2, biggest_add//2, ox, oy, angle)
    template_tr_rot = rotate_point(big_template.shape[1] - biggest_add//2, 0 + biggest_add//2, ox, oy, angle)
    template_bl_rot = rotate_point(0 + biggest_add//2, big_template.shape[0] - biggest_add//2, ox, oy, angle)
    template_br_rot = rotate_point(big_template.shape[1] - biggest_add//2, big_template.shape[0] - biggest_add//2, ox, oy, angle)

    return template_tl_rot, template_tr_rot, template_bl_rot, template_br_rot




def rotate_point(x, y, ox, oy, theta_deg):
        theta = np.deg2rad(theta_deg)
        dx, dy = x - ox, y - oy
        x_new = ox + dx * np.cos(theta) - dy * np.sin(theta)
        y_new = oy + dx * np.sin(theta) + dy * np.cos(theta)
        return x_new, y_new



def get_max_correlation(template_BW, maldi_o, flip, angle, angle_lo, angle_hi, step, meth):
    # Perform template matching
    maldi = rotate_image(maldi_o.copy(), angle)
    maldi[maldi == 0] = 255

    if flip:
        maldi = cv2.flip(maldi, 1)  # Flip the image horizontally
    if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        max_corr = float('inf')
    else: max_corr = 0
    max_angle = 0
    for ang in np.arange(angle_lo, angle_hi, step):
        maldi_rot = rotate_image(maldi.copy(), ang)
        maldi_rot[maldi_rot == 0] = 255
        result = cv2.matchTemplate(maldi_rot.copy(), template_BW.copy(), meth)
        min_val, max_val, _, _ = cv2.minMaxLoc(result)
        # print(f"Angle: {ang}, Max correlation: {max_val}")
        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if min_val < max_corr:
                max_corr = min_val
                max_angle = ang
        else:
            if max_val > max_corr:
                max_corr = max_val
                max_angle = ang
    return max_corr, max_angle


def process_images(sample_name, directory, template_pixel_size, maldi_pixel_size):
    # Load the images
    # template = cv2.imread('/home/eduard/Desktop/mortadela_samples/BU1627T2.png')  # Base image in color
    # template_BW = cv2.imread('/home/eduard/Desktop/mortadela_samples/BU1627T2.png', cv2.IMREAD_GRAYSCALE)  # Base image in grayscale
    # maldi_o = cv2.imread('/home/eduard/Desktop/mortadela_samples/BU1627T2_MALDI.png', cv2.IMREAD_GRAYSCALE)  # Overlay image in grayscale
    template_original = cv2.imread(f'{directory}/{sample_name}/{sample_name}.png')  # Base image in color
    template_BW_original = cv2.imread(f'{directory}/{sample_name}/{sample_name}.png', cv2.IMREAD_GRAYSCALE)  # Base image in grayscale
    maldi_o = cv2.imread(f'{directory}/{sample_name}/{sample_name}_TIC.png', cv2.IMREAD_GRAYSCALE)  # Overlay image in grayscale
    # plt.imshow(maldi_o, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # plt.close()


    ####### fusió de 3 maldi images ########
    # image1 = cv2.imread(f'{directory}/{sample_name}/{sample_name}_MAX.png')
    # image2 = cv2.imread(f'{directory}/{sample_name}/{sample_name}_RMS.png')
    # image3 = cv2.imread(f'{directory}/{sample_name}/{sample_name}_TIC.png')
    # fused_image = (image1.astype(np.float32) + image2.astype(np.float32) + image3.astype(np.float32)) / 3
    # fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
    # # convert fused_image to grayscaleBU204T1
    # fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2GRAY)



    image_paths = [
    f'{directory}/{sample_name}/{sample_name}_MAX.png',
    f'{directory}/{sample_name}/{sample_name}_RMS.png',
    f'{directory}/{sample_name}/{sample_name}_TIC.png'
    ]

    loaded_images = []
    for path in image_paths:
        if os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                loaded_images.append(img.astype(np.float32))

    if not loaded_images:
        raise ValueError("No images found to fuse")

    # Fuse by averaging only the loaded images
    fused_image = sum(loaded_images) / len(loaded_images)

    # Clip values and convert to uint8
    fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)

    # Convert to grayscale
    fused_image = cv2.cvtColor(fused_image, cv2.COLOR_BGR2GRAY)





    maldi_o = fused_image.copy()
    plt.imshow(maldi_o, cmap='gray')
    plt.axis('off')
    plt.show()
    # exit(1)

    # _, maldi_o = cv2.threshold(maldi_o.copy(), 0, 255, cv2.THRESH_OTSU)

    # plt.imshow(maldi_o, cmap='gray')
    # plt.axis('off')
    # plt.show()


    # template_pixel_size = 6.2568
    scaling_factor = template_pixel_size / maldi_pixel_size
    # scaling_factor = 0.8
    print(f"Scaling factor: {scaling_factor}")
    # plt.imshow(template, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # print(f"Template shape: {template.shape}")


    # print(f"Maldi shape: {maldi_o.shape}")
    # plt.imshow(template, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # print(f"Template shape: {template.shape}")


    # get maldi diagonal size
    maldi_diag = int(np.sqrt(maldi_o.shape[0]**2 + maldi_o.shape[1]**2))
    #add a border to the maldi image to make it square
    add_vertical = (maldi_diag - maldi_o.shape[0])
    add_horizontal = (maldi_diag - maldi_o.shape[1])
    #add half of the difference to each side



    # maldi_o = cv2.copyMakeBorder(maldi_o.copy(), add_vertical//2, add_vertical//2, add_horizontal//2, add_horizontal//2, cv2.BORDER_CONSTANT, value=255)
    # #add the biggest of add vertical and horizontal to the template, the padding should be white

    # biggest_add = max(add_vertical, add_horizontal)
    # print(f"biggest_add: {biggest_add}")
    biggest_add = int(maldi_diag / 2)
    print(f"biggest_add: {biggest_add}")

    # template = cv2.copyMakeBorder(template, biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    # template_BW = cv2.copyMakeBorder(template_BW, biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=255)



    # print(f"Template shape after padding: {template.shape}")
    # print(f"Maldi shape after padding: {maldi_o.shape}")
    # # #plot the image
    # plt.imshow(maldi_o, cmap='gray')
    # plt.axis('off')
    # plt.show()
    # exit(1)
    return template_original, template_BW_original, maldi_o, scaling_factor, biggest_add

def process_resize_chunk(method_name, template_original, template_BW_original, maldi_o, 
                        resize_values, biggest_add, top_picks):
    """
    Process a chunk of resize values for a given method
    """
    method = getattr(cv2, method_name)
    chunk_results = []
    
    for resize in resize_values:
        if resize <= 0:
            continue
            
        template = cv2.resize(template_original.copy(), (0, 0), fx=resize, fy=resize)
        template_BW = cv2.resize(template_BW_original.copy(), (0, 0), fx=resize, fy=resize)
        template = cv2.copyMakeBorder(template.copy(), biggest_add//2, biggest_add//2, 
                                    biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        template_BW = cv2.copyMakeBorder(template_BW.copy(), biggest_add//2, biggest_add//2, 
                                       biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=255)
        
        if template.shape[0] < maldi_o.shape[0] or template.shape[1] < maldi_o.shape[1]:
            continue
            
        max_corr_and_flip = []
        for angle in range(0, 360, 4):
            maldi = rotate_image(maldi_o.copy(), angle)
            maldi[maldi == 0] = 255
            maldi_flipped = cv2.flip(maldi.copy(), 1)

            # Apply template Matching
            res = cv2.matchTemplate(maldi.copy(), template_BW.copy(), method)
            res_flipped = cv2.matchTemplate(maldi_flipped.copy(), template_BW.copy(), method)
            min_val_flipped, max_val_flipped, _, _ = cv2.minMaxLoc(res_flipped)
            min_val, max_val, _, _ = cv2.minMaxLoc(res)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                if min_val_flipped < min_val:
                    max_corr_and_flip.append((min_val_flipped, angle, True))
                else:
                    max_corr_and_flip.append((min_val, angle, False))
            else:
                if max_val_flipped > max_val:
                    max_corr_and_flip.append((max_val_flipped, angle, True))
                else:
                    max_corr_and_flip.append((max_val, angle, False))

        # Sort and get top picks
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_picks_list = sorted(max_corr_and_flip, reverse=False)[:top_picks]
        else:
            top_picks_list = sorted(max_corr_and_flip, reverse=True)[:top_picks]
            
        # First refinement
        temp = []
        for i, (val, angle, flip) in enumerate(top_picks_list):
            corr, angle_var = get_max_correlation(template_BW.copy(), maldi_o.copy(), 
                                                flip, angle, -3, 3, 1, method)
            temp.append((corr, angle + angle_var, flip))
        top_picks_list = temp

        # Second refinement
        temp = []
        for i, (val, angle, flip) in enumerate(top_picks_list):
            corr, angle_var = get_max_correlation(template_BW.copy(), maldi_o.copy(), 
                                                flip, angle, -0.5, 0.1, 0.5, method)
            temp.append((corr, angle + angle_var, flip))
        top_picks_list = temp

        # Third refinement
        temp = []
        for i, (val, angle, flip) in enumerate(top_picks_list):
            corr, angle_var = get_max_correlation(template_BW.copy(), maldi_o.copy(), 
                                                flip, angle, -0.05, 0.01, 0.05, method)
            temp.append((corr, angle + angle_var, flip))
        top_picks_list = temp

        # Get best match
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            max_val, angle, flip = min(top_picks_list, key=lambda x: x[0])
        else:
            max_val, angle, flip = max(top_picks_list, key=lambda x: x[0])
            
        chunk_results.append((max_val, angle, flip, resize))
    
    return chunk_results


def create_overlay_gif(template, maldi, max_loc, output_path, alpha_steps=20, duration=100):
    """
    Create a GIF showing the overlay animation from alpha 0 to 0.7
    
    Args:
        template: Template image (BGR format)
        maldi: MALDI image (grayscale)
        max_loc: Location for overlay (x, y)
        output_path: Path to save the GIF
        alpha_steps: Number of frames in the animation (default: 20)
        duration: Duration of each frame in milliseconds (default: 100)
    """
    frames = []
    alpha_values = np.linspace(0, 0.7, alpha_steps)
    
    for alpha in alpha_values:
        # Create overlay for current alpha
        frame = overlay_images_alpha(template.copy(), maldi, max_loc, alpha)
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb.astype(np.uint8))
        frames.append(pil_frame)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Infinite loop
    )
    
    return frames

def overlay_images_alpha(template, maldi, max_loc, alpha):
    """
    Overlay images with specified alpha value
    """
    template_copy = template.copy()
    for c in range(0, 3):  # Loop over the color channels
        template_copy[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] = \
            (1 - alpha) * template_copy[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] + \
            alpha * maldi[:maldi.shape[0], :maldi.shape[1]] / 255.0 * 255  # Scale maldi to match color channels
    return template_copy


def create_overlay_gif_with_title2(maldi, output_path, alpha_steps=360, duration=100):
    """
    Create a GIF showing the overlay animation from alpha 0 to 0.7 with title on each frame
    
    Args:
        template: Template image (BGR format)
        maldi: MALDI image (grayscale)
        max_loc: Location for overlay (x, y)
        output_path: Path to save the GIF
        title_text: Title text to display on each frame
        alpha_steps: Number of frames in the animation (default: 20)
        duration: Duration of each frame in milliseconds (default: 100)
    """
    frames = []
    
    for alpha in range(0, alpha_steps):
        # Create overlay for current alpha
        test = rotate_image(maldi.copy(), alpha)
        plt.imshow(test, cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()
        # print(f"dimensionas test: {test.shape}")

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb.astype(np.uint8))
        
        frames.append(pil_frame)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Infinite loop
    )
    # exit(1)
    
    return frames


def create_overlay_gif_with_title(template, maldi, max_loc, output_path, title_text, alpha_steps=20, duration=100):
    """
    Create a GIF showing the overlay animation from alpha 0 to 0.7 with title on each frame
    
    Args:
        template: Template image (BGR format)
        maldi: MALDI image (grayscale)
        max_loc: Location for overlay (x, y)
        output_path: Path to save the GIF
        title_text: Title text to display on each frame
        alpha_steps: Number of frames in the animation (default: 20)
        duration: Duration of each frame in milliseconds (default: 100)
    """
    frames = []
    alpha_values = np.linspace(0, 0.7, alpha_steps)
    
    for alpha in alpha_values:
        # Create overlay for current alpha
        frame = overlay_images_alpha(template.copy(), maldi, max_loc, alpha)
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_frame = Image.fromarray(frame_rgb.astype(np.uint8))
        
        # Add title to the frame
        pil_frame_with_title = add_title_to_image(pil_frame, title_text)
        frames.append(pil_frame_with_title)
    
    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Infinite loop
    )
    
    return frames

def add_title_to_image(image, title_text, title_height=100, font_size=20):
    """
    Add title text at the top of an image
    
    Args:
        image: PIL Image
        title_text: Text to display
        title_height: Height of title area in pixels
        font_size: Font size for title
    """
    # Get image dimensions
    width, height = image.size
    
    # Create new image with extra height for title
    new_image = Image.new('RGB', (width, height + title_height), color='white')
    
    # Paste original image below title area
    new_image.paste(image, (0, title_height))
    
    # Draw title
    draw = ImageDraw.Draw(new_image)
    
    try:
        # Try to use a better font if available - THIS IS WHERE FONT SIZE IS SET
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            # Try other common font paths
            font = ImageFont.truetype("Arial.ttf", font_size)  # Windows
        except:
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)  # macOS
            except:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)  # Linux
                except:
                    # Fall back to default font (this will be small)
                    font = ImageFont.load_default()
                    print(f"Warning: Could not load TrueType font. Using default font.")
    
    # Get text dimensions for centering
    bbox = draw.textbbox((0, 0), title_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Calculate position to center text
    x = (width - text_width) // 2
    y = (title_height - text_height) // 2
    
    # Draw text - REMOVE font_size parameter, it doesn't exist in draw.text()
    draw.text(xy=(x, y), text=title_text, fill='black', font=font)
    
    return new_image



def match_and_overlay_gif_and_png(template_original, template_BW_original, maldi_o, scaling_factor, biggest_add, sample_name, directory, top_picks, threads=1, alpha_steps=20, frame_duration=100):
    """
    Modified version that creates both GIFs and final PNG files
    """
    methods = ['TM_CCOEFF', 
               'TM_CCOEFF_NORMED', # el millor
            #    'TM_CCORR',
               'TM_CCORR_NORMED', 
               'TM_SQDIFF', #va raro x2
               'TM_SQDIFF_NORMED'] # va raro

    min_range = scaling_factor - 0.2
    if min_range <= 0:
        min_range = 0.05

    for meth in methods:
        method = getattr(cv2, meth)
        print(f"Processing method: {meth}")
        
        # [Same threading logic as before...]
        resize_values = np.arange(min_range, scaling_factor + 0.2, 0.005)
        chunk_size = max(1, len(resize_values) // threads)
        
        resize_chunks = []
        for i in range(0, len(resize_values), chunk_size):
            chunk = resize_values[i:i + chunk_size]
            if len(chunk) > 0:
                resize_chunks.append(chunk)
        
        top_values = []
        with ThreadPoolExecutor(max_workers=threads) as executor:
            future_to_chunk = {
                executor.submit(process_resize_chunk, meth, template_original, 
                              template_BW_original, maldi_o, chunk, biggest_add, top_picks): chunk 
                for chunk in resize_chunks
            }
            
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                top_values.extend(chunk_results)
        
        top_values.sort(key=lambda x: x[3])
        
        if len(top_values) == 0:
            print(f"No valid template sizes found for method {meth}. Skipping.")
            continue
            
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            max_val, angle, flip, resize = min(top_values, key=lambda x: x[0])
        else:
            max_val, angle, flip, resize = max(top_values, key=lambda x: x[0])

        # [Same image processing as before...]
        template = cv2.resize(template_original.copy(), (0, 0), fx=resize, fy=resize)
        template_BW = cv2.resize(template_BW_original.copy(), (0, 0), fx=resize, fy=resize)
        template = cv2.copyMakeBorder(template.copy(), biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        template_BW = cv2.copyMakeBorder(template_BW.copy(), biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=255)

        maldi = rotate_image(maldi_o.copy(), angle)
        maldi[maldi == 0] = 255
        if flip:
            maldi = cv2.flip(maldi.copy(), 1)
        res = cv2.matchTemplate(maldi.copy(), template_BW.copy(), method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            overlay_loc = min_loc
        else:
            overlay_loc = max_loc

        os.makedirs(f"{directory}/outputs/{sample_name}", exist_ok=True)
        os.makedirs(f"{directory}/outputs/{sample_name}/gifs", exist_ok=True)

        # Create GIF animation
        gif_path = f"{directory}/outputs/{sample_name}/gifs/{sample_name}_{meth}.gif"
        create_overlay_gif_with_title(template.copy(), maldi.copy(), overlay_loc, gif_path, f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'",
                          alpha_steps=alpha_steps, duration=frame_duration)
        

        # Also create final PNG with alpha=0.7 (optional)
        template_with_overlay1 = overlay_images_alpha(template.copy(), maldi.copy(), overlay_loc, 0.7)
        # plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(template_with_overlay1, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'")
        plt.savefig(f"{directory}/outputs/{sample_name}/{sample_name}_{meth}.png", bbox_inches='tight')
        plt.show()
        plt.close()  # Important: close the figure to free memory
        
        print(f"{meth} done - GIF and PNG created")






        template_with_overlay1 = overlay_images_alpha(template.copy(), maldi.copy(), overlay_loc, 0.7)
        # plt.figure(figsize=(10, 8))
        # plt.imshow(cv2.cvtColor(template_with_overlay1, cv2.COLOR_BGR2RGB), aspect='equal')
        # plt.axis('off')
        # plt.title(f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'")
        # plt.savefig(f"{directory}/outputs/{sample_name}_30/{sample_name}_{meth}.png", bbox_inches='tight')
        #put a red dot on overlay_loc


        #4 points maldi
        # maldi_top_left = overlay_loc
        # maldi_top_right = (overlay_loc[0] + maldi.shape[1], overlay_loc[1])
        # maldi_bottom_left = (overlay_loc[0], overlay_loc[1] + maldi.shape[0])
        # maldi_bottom_right = (overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0]) 
        # plt.plot([maldi_top_left[0], maldi_top_right[0]], [maldi_top_left[1], maldi_top_right[1]], 'r-')  # Top edge
        # plt.plot([maldi_top_left[0], maldi_bottom_left[0]], [maldi_top_left[1], maldi_bottom_left[1]], 'r-')  # Left edge
        # plt.plot([maldi_bottom_left[0], maldi_bottom_right[0]], [maldi_bottom_left[1], maldi_bottom_right[1]], 'r-')  # Bottom edge     
        # plt.plot([maldi_top_right[0], maldi_bottom_right[0]], [maldi_top_right[1], maldi_bottom_right[1]], 'r-')  # Right edge  

        maldi_o_rot = rotate_image(maldi_o.copy(), angle)
        maldi_x_displ = (maldi_o_rot.shape[0] - maldi_o.shape[0]) // 2
        maldi_y_displ = (maldi_o_rot.shape[1] - maldi_o.shape[1]) // 2

        # plt.plot(overlay_loc[0] + maldi_x_displ, overlay_loc[1] + maldi_y_displ, 'ro')  #maldi top left
        # plt.plot(template_with_overlay1.shape[0] - overlay_loc[0]- maldi_x_displ, overlay_loc[1]+ maldi_y_displ, 'ro')  #
        # plt.plot([0, template_with_overlay1.shape[0]], [0, 0], 'r-')
        # plt.plot([0, 0], [0, template_with_overlay1.shape[1]], 'r-')
        # plt.plot([template_with_overlay1.shape[0], template_with_overlay1.shape[0]], [0, template_with_overlay1.shape[1]], 'r-')
        # plt.plot([0, template_with_overlay1.shape[0]], [template_with_overlay1.shape[1], template_with_overlay1.shape[1]], 'r-')

        # plt.show()

        # plt.close()  # Important: close the figure to free memory
        
        # print(f"{meth} done - GIF and PNG created")

        # template_with_overlay1 = overlay_images_alpha(maldi.copy(), template.copy(), overlay_loc, 0.7)
        

        # print( f"overlay_loc[0]: {overlay_loc[0]}, overlay_loc[1]: {overlay_loc[1]}, template.shape[1]//2: {template.shape[1]//2}, template.shape[0]//2: {template.shape[0]//2}, angle: {angle}, flip: {flip}" )

        # if flip:
        #     x, y = rotate_point_manual(template_with_overlay1.shape[0] - overlay_loc[0] - biggest_add//2 - maldi_x_displ, overlay_loc[1]+ maldi_y_displ,
        #                         template_with_overlay1.shape[0]//2, template_with_overlay1.shape[1]//2, angle)
        # else:
        #     x, y = rotate_point(overlay_loc[0], overlay_loc[1], template.shape[0]//2, template.shape[1]//2, angle)
        template_with_overlay1_copy = template_with_overlay1.copy()
        template_with_overlay1 = remove_border(template_with_overlay1, biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2)

        if flip:
            test = cv2.flip(template_with_overlay1.copy(), 1)
            template_x_displ = (template_with_overlay1.shape[0] - test.shape[0]) // 2
            template_y_displ = (template_with_overlay1.shape[1] - test.shape[1]) // 2
            # plt.figure(figsize=(10, 8))
            # plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB), aspect='equal')
            # plt.axis('off')
            # plt.plot(test.shape[0] - overlay_loc[0], overlay_loc[1]+ maldi_y_displ -biggest_add//2, 'ro')
            # # plt.title(f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'")
            # # plt.savefig(f"{directory}/outputs/{sample_name}_30/{sample_name}_{meth}.png", bbox_inches='tight')
            # # plt.plot(x, y, 'ro')  #
            # plt.plot([0, template_with_overlay1.shape[0]], [0, 0], 'r-')
            # plt.plot([0, 0], [0, template_with_overlay1.shape[1]], 'r-')
            # plt.plot([template_with_overlay1.shape[0], template_with_overlay1.shape[0]], [0, template_with_overlay1.shape[1]], 'r-')
            # plt.plot([0, template_with_overlay1.shape[0]], [template_with_overlay1.shape[1], template_with_overlay1.shape[1]], 'r-')

            # #do the same for the inner maldi
            # # plt.plot([overlay_loc[0], overlay_loc[1]], [overlay_loc[0] + maldi.shape[0], overlay_loc[1]], 'b-')  # Top edge
            # # plt.plot([overlay_loc[0], overlay_loc[1]], [overlay_loc[0], overlay_loc[1] + maldi.shape[1]], 'b-')  # Left edge
            # # plt.plot([overlay_loc[0], overlay_loc[1] + maldi.shape[1]], [overlay_loc[0] + maldi.shape[0], overlay_loc[1] + maldi.shape[1]], 'b-')  # Bottom edge     
            # # plt.plot([overlay_loc[0] + maldi.shape[0], overlay_loc[1]], [overlay_loc[0] + maldi.shape[0], overlay_loc[1] + maldi.shape[ 1]], 'b-')  # Right edge
            # plt.show()
            # plt.close()  # Im
        else:
            test = template_with_overlay1.copy()
        # test1 = rotate_image(test.copy(), -angle)
        # print(f"maldi_o shape[0]: {maldi_o.shape[0]}, maldi_o shape[1]: {maldi_o.shape[1]}")
        # print(f"maldi_o_rot shape[0]: {maldi_o_rot.shape[0]}, maldi_o_rot shape[1]: {maldi_o_rot.shape[1]}")
        maldi_tl = (test.shape[0] - overlay_loc[0], overlay_loc[1]+ maldi_y_displ -biggest_add//2)
        
        #get points relative to maldi
        template_tl, template_tr, template_bl, template_br, maldi_tl_rot, = calculate_coordinates_if_maldi_topleft_is_00(maldi_tl, test.copy(), angle)
        #plot the points in a blank image, blue dots for template, red for maldi
        test1 = np.ones((test.shape[1]*2, test.shape[0]*2 , 3), dtype=np.uint8) * 255
        x = maldi_tl_rot[0], maldi_tl_rot[1]
        y = maldi_tl_rot[0], maldi_tl_rot[1]
        # plt.imshow(test1)
        # plt.axis('off')
        # plt.plot(maldi_tl_rot[0], maldi_tl_rot[1], 'ro')  #maldi top left
        # plt.plot(maldi_tl_rot[0] + maldi_o.shape[0], maldi_tl_rot[1], 'ro')  #
        # plt.plot(maldi_tl_rot[0], maldi_tl_rot[1] + maldi_o.shape[1], 'ro')  #
        # plt.plot(maldi_tl_rot[0] + maldi_o.shape[0], maldi_tl_rot[1] + maldi_o.shape[1], 'ro')

        # plt.plot(template_tl[0], template_tl[1], 'bo')  #template top left
        # plt.plot(template_tr[0], template_tr[1], 'bo')  #template top right
        # plt.plot(template_bl[0], template_bl[1], 'bo')  #template bottom left
        # plt.plot(template_br[0], template_br[1], 'bo')  #template bottom right
        # # plt.title(f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'")
        # plt.show()
        # plt.close()  # Important: close the figure to free memory
        
        
        test = rotate_image(test.copy(), -angle)
        # print(f"test shape[0]: {test.shape[0]}, test shape[1]: {test.shape[1]}")
        # plt.figure(figsize=(8, 10))
        # plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))#, aspect='equal')
        # plt.axis('off')
        # # plt.gca().set_aspect('equal')
        # # plt.title(f"method: {meth}, angle: {angle}, flip: {flip}, resize: {resize:.3f}, location: {overlay_loc}'")
        # # plt.savefig(f"{directory}/outputs/{sample_name}_30/{sample_name}_{meth}.png", bbox_inches='tight')
        # # plt.plot(x, y, 'ro')  #
        # # plt.plot([0, test.shape[0]], [0, 0], 'r-')
        # # plt.plot([0, 0], [0, test.shape[1]], 'r-')
        # # plt.plot([test.shape[0], test.shape[0]], [0, test.shape[1]], 'r-')
        # # plt.plot([0, test.shape[0]], [test.shape[1], test.shape[1]], 'r-')
        # plt.show()
        # plt.close()  # Important: close the figure to free memory


        # print("la bona")
        maldi = maldi_o.copy()

        maldi[maldi == 0] = 255
        print("Starting super fine tuning...")
        print(f"Initial resize: {resize}")
        final_resize_table = []
        min_resize = resize - 0.02
        max_resize = resize + 0.02
        if min_resize <= 0:
            min_resize = 0.05

        for resize_final in np.arange(min_resize, max_resize, 0.005):
            # template_resized = cv2.resize(template.copy(), (0, 0), fx=resize_final, fy=resize_final)
            template_BW_resized = cv2.resize(template_BW_original.copy(), (0, 0), fx=resize_final, fy=resize_final)
            template_BW_resized = cv2.copyMakeBorder(template_BW_resized.copy(), biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=255)
            template_BW_resized = rotate_image(template_BW_resized.copy(), angle)
            template_BW_resized = cv2.flip(template_BW_resized.copy(), 1) if flip else template_BW_resized.copy()

            if template_BW_resized.shape[0] < maldi.shape[0] or template_BW_resized.shape[1] < maldi.shape[1]:
                continue
            # template_with_overlay2 = overlay_images_alpha(template_resized.copy(), maldi.copy(), overlay_loc, 0.7)
            
            res = cv2.matchTemplate(maldi.copy(), template_BW_resized.copy(), method)
            min_val, max_val, min_loc, max_loc  = cv2.minMaxLoc(res)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                final_resize_table.append((min_val, resize_final, min_loc))
            else:            
                # print(f"max val: {max_val}, max_loc: {max_loc} for resize_final: {resize_final}")
                final_resize_table.append((max_val, resize_final, max_loc))
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            max_val, resize, location = min(final_resize_table, key=lambda x: x[0])
        else:
            max_val, resize, location = max(final_resize_table, key=lambda x: x[0])


        overlay_loc = location
        print(f"Final resize: {resize}")
        print(f"Final location: {overlay_loc}")

        template = cv2.resize(template_original.copy(), (0, 0), fx=resize, fy=resize)
        template = cv2.copyMakeBorder(template.copy(), biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=255)
 
        template = rotate_image(template.copy(), angle)

        template = cv2.flip(template.copy(), 1) if flip else template.copy()
 
        template_with_overlay2 = overlay_images_alpha(template.copy(), maldi.copy(), overlay_loc, 0.7)

        
        # plt.plot(overlay_loc[0], overlay_loc[1], 'ro')  #maldi top left
        # plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1], 'ro')  #
        # plt.plot(overlay_loc[0], overlay_loc[1] + maldi.shape[0], 'ro')  #
        # plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0], 'ro')
        # maldi top line
        # plt.plot([overlay_loc[0], overlay_loc[1]], [overlay_loc[0] + maldi.shape[1], overlay_loc[1]], 'b-')  # Top edge
        # plt.plot([overlay_loc[0], overlay_loc[1]], [overlay_loc[0], overlay_loc[1] + maldi.shape[0]], 'b-')  # Left edge
        # plt.plot([overlay_loc[0], overlay_loc[1] + maldi.shape[0]], [overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0]], 'b-')  # Bottom edge     
        # plt.plot([overlay_loc[0] + maldi.shape[1], overlay_loc[1]], [overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0]], 'b-')  # Right edge
        #plot template points minus the borders taking into account the rotation and flip
        maldi_tl = (overlay_loc[0], overlay_loc[1])
        template_tl, template_tr, template_bl, template_br, _, = calculate_coordinates_if_maldi_topleft_is_00(maldi_tl, template_with_overlay2.copy(), angle)
        template_tl_o, template_tr_o, template_bl_o, template_br_o, = calculate_coordinates_if_maldi_topleft_is_00_final(template_with_overlay1_copy.copy(), template_original.copy(), angle, biggest_add)
        
        
        plt.imshow(cv2.cvtColor(template_with_overlay2, cv2.COLOR_BGR2RGB), aspect='equal')
        plt.axis('off')

        plt.plot(overlay_loc[0], overlay_loc[1], 'ro')  #maldi top left
        plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1], 'ro')  #
        plt.plot(overlay_loc[0], overlay_loc[1] + maldi.shape[0], 'ro')  #
        plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0], 'ro')

        plt.plot(template_tl_o[0], template_tl_o[1], 'wo')  #template top left
        plt.plot(template_tr_o[0], template_tr_o[1], 'go')  #template top right
        plt.plot(template_bl_o[0], template_bl_o[1], 'yo')  #template bottom left
        plt.plot(template_br_o[0], template_br_o[1], 'ko')  #template bottom right
        # plt.plot(0,0, 'go')  #origin
        # plt.plot([0, template_with_overlay1.shape[0]], [0, 0], 'r-')
        # plt.plot([0, 0], [0, template_with_overlay1.shape[1]], 'r-')
        # plt.plot([template_with_overlay1.shape[0], template_with_overlay1.shape[0]], [0, template_with_overlay1.shape[1]], 'r-')
        # plt.plot([0, template_with_overlay1.shape[0]], [template_with_overlay1.shape[1], template_with_overlay1.shape[1]], 'r-')

        maldi_tl_final = (overlay_loc[0], overlay_loc[1])
        maldi_tr_final = (overlay_loc[0] + maldi.shape[1], overlay_loc[1])
        maldi_bl_final = (overlay_loc[0], overlay_loc[1] + maldi.shape[0])
        maldi_br_final = (overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0])

        transform_coordinates(maldi_tl_final, maldi_tr_final, maldi_bl_final, maldi_br_final,
                              template_tl_o, template_tr_o, template_bl_o, template_br_o)


        plt.show()

        plt.close()  # Important: close the figure to free memory





        # exit(0)


# def rotate_point_manual(x, y, center_x, center_y, angle_degrees, scale=1.0):
#     """
#     Manually rotate a point around a center using the same convention as OpenCV.
    
#     Args:
#         x, y: Original point coordinates
#         center_x, center_y: Center of rotation
#         angle_degrees: Rotation angle in degrees (positive = counterclockwise in image coords)
#         scale: Scale factor (usually 1.0)
    
#     Returns:
#         x_rotated, y_rotated: New coordinates after rotation
#     """
#     # Convert angle to radians
#     angle_rad = np.radians(angle_degrees)
    
#     # Calculate rotation components
#     cos_angle = np.cos(angle_rad) * scale
#     sin_angle = np.sin(angle_rad) * scale
    
#     # Translate point to origin (relative to center)
#     dx = x - center_x
#     dy = y - center_y
    
#     # Apply rotation
#     x_rotated = center_x + cos_angle * dx - sin_angle * dy
#     y_rotated = center_y + sin_angle * dx + cos_angle * dy
    
#     return x_rotated, y_rotated


def remove_border(image, top, bottom, left, right):
    """
    Remove border by cropping the image using array slicing
    
    Args:
        image: Input image
        top, bottom, left, right: Border sizes to remove from each side
    
    Returns:
        Cropped image without borders
    """
    height, width = image.shape[:2]
    
    # Crop using array slicing
    cropped = image[top:height-bottom, left:width-right]
    
    return cropped

def transform_coordinates(maldi_tl, maldi_tr, maldi_bl, maldi_br,
                          template_tl, template_tr, template_bl, template_br):
    """
    Print the transformation coordinates taking into account maldi TL is (0,0)
    """
    maldi_tl_new = (0, 0)
    maldi_tr_new = (maldi_tr[0] - maldi_tl[0], 0)
    maldi_bl_new = (0, maldi_bl[1] - maldi_tl[1])
    maldi_br_new = (maldi_br[0] - maldi_bl[0], maldi_br[1] - maldi_tr[1])
    # template_tl_
    print("Transformation Coordinates:")
 