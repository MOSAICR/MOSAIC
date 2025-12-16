import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageDraw, ImageFont

# Read pixel size from info file
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


# rotate image by angle in degrees
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



def calculate_template_rotated_coordinates(big_template, angle, biggest_add):
    """
    Calculate the rotated coordinates of the template corners after rotation.
    Args:
        big_template: The padded template image.
        angle: Rotation angle in degrees.
        biggest_add: The size of the padding added to the template.
    Returns:
        A tuple containing the rotated coordinates of the template corners:
        (top-left, top-right, bottom-left, bottom-right)
    """


    ox, oy = big_template.shape[1] // 2, big_template.shape[0] // 2


    template_tl_rot = rotate_point(biggest_add//2, biggest_add//2, ox, oy, angle)
    template_tr_rot = rotate_point(big_template.shape[1] - biggest_add//2, 0 + biggest_add//2, ox, oy, angle)
    template_bl_rot = rotate_point(0 + biggest_add//2, big_template.shape[0] - biggest_add//2, ox, oy, angle)
    template_br_rot = rotate_point(big_template.shape[1] - biggest_add//2, big_template.shape[0] - biggest_add//2, ox, oy, angle)

    return template_tl_rot, template_tr_rot, template_bl_rot, template_br_rot



# roatate a point given the image center and angle in degrees
def rotate_point(x, y, ox, oy, theta_deg):
        theta = np.deg2rad(theta_deg)
        dx, dy = x - ox, y - oy
        x_new = ox + dx * np.cos(theta) - dy * np.sin(theta)
        y_new = oy + dx * np.sin(theta) + dy * np.cos(theta)
        return x_new, y_new


# get max correlation with angle refinement
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
        if meth in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            if min_val < max_corr:
                max_corr = min_val
                max_angle = ang
        else:
            if max_val > max_corr:
                max_corr = max_val
                max_angle = ang
    return max_corr, max_angle


# Process images and return necessary variables
def process_images(sample_name, directory, template_pixel_size, maldi_pixel_size):
    template_original = cv2.imread(f'{directory}/{sample_name}/{sample_name}.png')  # Base image in color
    template_BW_original = cv2.imread(f'{directory}/{sample_name}/{sample_name}.png', cv2.IMREAD_GRAYSCALE)  # Base image in grayscale
    maldi_o = cv2.imread(f'{directory}/{sample_name}/{sample_name}_TIC.png', cv2.IMREAD_GRAYSCALE)  # Overlay image in grayscale


    # fusing 3 maldi images (if there are less, it still fuses them)
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
   
    scaling_factor = template_pixel_size / maldi_pixel_size
    # print(f"Scaling factor: {scaling_factor}")

    # get maldi diagonal size
    maldi_diag = int(np.sqrt(maldi_o.shape[0]**2 + maldi_o.shape[1]**2))
    
    biggest_add = int(maldi_diag / 2)
   
    return template_original, template_BW_original, maldi_o, scaling_factor, biggest_add


# Process a chunk of resize values for a given method
def process_resize_chunk(method_name, template_original, template_BW_original, maldi_o, 
                        resize_values, biggest_add, top_picks):
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
        
        # Skip if resized template is smaller than maldi
        if template.shape[0] < maldi_o.shape[0] or template.shape[1] < maldi_o.shape[1]:
            continue
            
        max_corr_and_flip = []
        # Initial scan over angles, big steps
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


# Overlay template and maldi with specified alpha value at the top correlation location
def overlay_images_alpha(template, maldi, max_loc, alpha):
    template_copy = template.copy()
    for c in range(0, 3):  # Loop over the color channels
        template_copy[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] = \
            (1 - alpha) * template_copy[max_loc[1]:max_loc[1] + maldi.shape[0], max_loc[0]:max_loc[0] + maldi.shape[1], c] + \
            alpha * maldi[:maldi.shape[0], :maldi.shape[1]] / 255.0 * 255  # Scale maldi to match color channels
    return template_copy



def create_overlay_gif(template, maldi, max_loc, output_path, alpha_steps=20, duration=100):
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


def match_and_overlay_gif_and_png(template_original, template_BW_original, maldi_o, scaling_factor, biggest_add, sample_name, directory, top_picks, threads=1, alpha_steps=20, frame_duration=100):
    """
    Perform template matching using multiple methods, create overlay GIFs and PNGs.
    Args:
        template_original: Original template image (color)
        template_BW_original: Original template image (grayscale)
        maldi_o: MALDI image (grayscale)
        scaling_factor: Scaling factor for resizing the template
        biggest_add: Padding size for the template
        sample_name: Sample name for output files
        directory: Output directory
        top_picks: Number of top picks to consider during matching
        threads: Number of threads for parallel processing
        alpha_steps: Number of frames in the GIF animation
        frame_duration: Duration of each frame in the GIF in milliseconds
    """
    # Define template matching methods to use
    methods = [#'TM_CCOEFF', 
               'TM_CCOEFF_NORMED']#, # el millor
            #    'TM_CCORR',
            #    'TM_CCORR_NORMED', 
            #    'TM_SQDIFF', #va raro x2
            #    'TM_SQDIFF_NORMED'] # va raro

    min_range = scaling_factor - 0.2
    if min_range <= 0:
        min_range = 0.05

    for meth in methods:
        method = getattr(cv2, meth)
        print(f"Processing method: {meth}")
        
        # threading logic
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

        # Get best match from top values    
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            max_val, angle, flip, resize = min(top_values, key=lambda x: x[0])
        else:
            max_val, angle, flip, resize = max(top_values, key=lambda x: x[0])



        # before here, template was on its original rotation and not flipped
        # after here, template is rotated and flipped as per best match

        maldi = maldi_o.copy()

        maldi[maldi == 0] = 255
        print("Starting super fine tuning...")
        # print(f"Initial resize: {resize}")
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
            
            res = cv2.matchTemplate(maldi.copy(), template_BW_resized.copy(), method)
            min_val, max_val, min_loc, max_loc  = cv2.minMaxLoc(res)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                final_resize_table.append((min_val, resize_final, min_loc))
            else:            
                final_resize_table.append((max_val, resize_final, max_loc))
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            max_val, resize, location = min(final_resize_table, key=lambda x: x[0])
        else:
            max_val, resize, location = max(final_resize_table, key=lambda x: x[0])


        overlay_loc = location
        # print(f"Final resize: {resize}")
        # print(f"Final location: {overlay_loc}")

        template = cv2.resize(template_original.copy(), (0, 0), fx=resize, fy=resize)
        template = cv2.copyMakeBorder(template.copy(), biggest_add//2, biggest_add//2, biggest_add//2, biggest_add//2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
 
        template = rotate_image(template.copy(), angle)

        template = cv2.flip(template.copy(), 1) if flip else template.copy()
 
        template_with_overlay2 = overlay_images_alpha(template.copy(), maldi.copy(), overlay_loc, 0.7)


        os.makedirs(f"{directory}/outputs/{sample_name}", exist_ok=True)
        os.makedirs(f"{directory}/outputs/{sample_name}/gifs", exist_ok=True)

        # Create GIF animation
        gif_path = f"{directory}/outputs/{sample_name}/gifs/{sample_name}_{meth}.gif"
        create_overlay_gif(template.copy(), maldi.copy(), overlay_loc, gif_path,
                          alpha_steps=alpha_steps, duration=frame_duration)
        

        # Also create final PNG with alpha=0.7 (optional)
        # plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(template_with_overlay2, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(f"sample: {sample_name}, method: {meth}, \nangle: {angle}, flip: {flip}, resize: {resize:.3f}")
        plt.savefig(f"{directory}/outputs/{sample_name}/{sample_name}_{meth}.png", bbox_inches='tight')
        
        print(f"{meth} done - GIF and PNG created")

    
        #plot template points minus the borders taking into account the rotation and flip
        maldi_tl = (overlay_loc[0], overlay_loc[1])
        template_tl_o, template_tr_o, template_bl_o, template_br_o, = calculate_template_rotated_coordinates(template_with_overlay2.copy(), angle, biggest_add)
        


        plt.plot(overlay_loc[0], overlay_loc[1], 'ro')  #maldi top left
        plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1], 'ro')  #
        plt.plot(overlay_loc[0], overlay_loc[1] + maldi.shape[0], 'ro')  #
        plt.plot(overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0], 'ro')

        plt.plot(template_tl_o[0], template_tl_o[1], 'bo')  #template top left
        plt.plot(template_tr_o[0], template_tr_o[1], 'go')  #template top right
        plt.plot(template_bl_o[0], template_bl_o[1], 'yo')  #template bottom left
        plt.plot(template_br_o[0], template_br_o[1], 'ko')  #template bottom right


        maldi_tl_final = (overlay_loc[0], overlay_loc[1])
        maldi_tr_final = (overlay_loc[0] + maldi.shape[1], overlay_loc[1])
        maldi_bl_final = (overlay_loc[0], overlay_loc[1] + maldi.shape[0])
        maldi_br_final = (overlay_loc[0] + maldi.shape[1], overlay_loc[1] + maldi.shape[0])

        transform_coordinates(maldi_tl_final, maldi_tr_final, maldi_bl_final, maldi_br_final,
                              template_tl_o, template_tr_o, template_bl_o, template_br_o, 
                              directory, sample_name, meth, angle, flip, resize)


        plt.show()

        plt.close()  # Important: close the figure to free memory



def transform_coordinates(maldi_tl, maldi_tr, maldi_bl, maldi_br,
                          template_tl, template_tr, template_bl, 
                          template_br, directory, sample_name, meth, angle, flip, resize):
    """
    Print the transformation coordinates taking into account maldi TL is (0,0)
    """
    maldi_tl_new = (0, 0)
    maldi_tr_new = (maldi_tr[0] - maldi_tl[0], 0)
    maldi_bl_new = (0, maldi_bl[1] - maldi_tl[1])
    maldi_br_new = (maldi_br[0] - maldi_bl[0], maldi_br[1] - maldi_tr[1])

    template_tl_new = (template_tl[0] - maldi_tl[0], template_tl[1] - maldi_tl[1])
    template_tr_new = (template_tr[0] - maldi_tl[0], template_tr[1] - maldi_tl[1])
    template_bl_new = (template_bl[0] - maldi_tl[0], template_bl[1] - maldi_tl[1])
    template_br_new = (template_br[0] - maldi_tl[0], template_br[1] - maldi_tl[1])
    
    os.makedirs(f"{directory}/outputs/{sample_name}/coordinates", exist_ok=True)
    txt_path_maldi = f"{directory}/outputs/{sample_name}/coordinates/{sample_name}_{meth}_MALDI.txt"
    txt_path_qupath = f"{directory}/outputs/{sample_name}/coordinates/{sample_name}_{meth}_QUPATH.txt"

    with open(txt_path_maldi, 'w') as f:
        f.write(f"TL {maldi_tl_new[0]} {maldi_tl_new[1]}\n")
        f.write(f"TR {maldi_tr_new[0]} {maldi_tr_new[1]}\n")
        f.write(f"BL {maldi_bl_new[0]} {maldi_bl_new[1]}\n")
        f.write(f"BR {maldi_br_new[0]} {maldi_br_new[1]}\n")
    f.close()
    with open(txt_path_qupath, 'w') as f:
        f.write(f"angle: {angle}\n")
        f.write(f"flip: {flip}\n")
        f.write(f"resize: {resize}\n")
        f.write(f"TL {template_tl_new[0]} {template_tl_new[1]}\n")
        f.write(f"TR {template_tr_new[0]} {template_tr_new[1]}\n")
        f.write(f"BL {template_bl_new[0]} {template_bl_new[1]}\n")
        f.write(f"BR {template_br_new[0]} {template_br_new[1]}\n")
    f.close()


    
 