import fitz  # PyMuPDF
import re
import cv2
import numpy as np
import streamlit as st
from io import BytesIO

# Function to find specific details using regular expressions
def find_detail(text, pattern, group=1):
    match = re.search(pattern, text)
    return match.group(group) if match else "Not found"

# Function to find white text and its bounding box
def find_white_text(page):
    text_instances = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Checking if the text color is white (1.0 for R, G, B in PDF context)
                    if span["color"] == 1.0:
                        text_instances.append({
                            "text": span["text"],
                            "bbox": span["bbox"]
                        })
    return text_instances

# Function to extract bounding boxes for specific text
def extract_bounding_boxes(page, keyword):
    bounding_boxes = []
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if keyword in span["text"]:
                        bounding_boxes.append(span["bbox"])
    return bounding_boxes

st.title("GAS PIPELINES IDENTIFICATION APP")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Read the PDF file
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    filename = uploaded_file.name

    # Extract the first page of the PDF and convert it to an image
    page = pdf_document.load_page(0)  # load the first page
    pix = page.get_pixmap()
    image_path = "image.png"
    pix.save(image_path)

    # Extract text from the first page
    text = page.get_text()

    # Patterns to search for specific details
    patterns = {
        "Date Requested": r'Date Requested:\s*(\S+)',
        "Job Reference": r'Job Reference:\s*(\S+)',
        "Site Location": r'Site Location:\s*([\d\s]+)',
        "Your Scheme/Reference": r'Your Scheme/Reference:\s*(\S+)',
        "Gas Warning": r'WARNING! This area contains (.*)'
    }

    # Extract and display the details
    details = {key: find_detail(text, pattern) for key, pattern in patterns.items()}
    st.write("\nExtracted Details:")
    st.write(f"Filename: {filename}")
    for key, value in details.items():
        st.write(f"{key}: {value}")

    # Find and display white text
    white_texts = find_white_text(page)
    for white_text in white_texts:
        st.write(white_text["text"])

    # Extract bounding boxes for the phrase "RISK OF DEATH OR SERIOUS INJURY"
    caution_boxes = extract_bounding_boxes(page, "RISK OF DEATH OR SERIOUS INJURY")

    # Check if the phrase "RISK OF DEATH OR SERIOUS INJURY" is found
    if caution_boxes:
        st.write("GAS 'WALES & WEST': Found")
        for bbox in caution_boxes:
            x_min, y_min, x_max, y_max = map(int, bbox)
            # st.write(f"Bounding Box: {bbox}")
            # Optionally draw bounding boxes or display the image as before
    else:
        st.write("GAS 'WALES & WEST': Not Found")

    # Load the image
    img = cv2.imread(image_path)

    # Draw bounding boxes around white text
    for white_text in white_texts:
        x_min, y_min, x_max, y_max = map(int, white_text["bbox"])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

    # Draw bounding boxes around "EXTREME CAUTION" text
    for bbox in caution_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue bounding box

    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Convert the RGB color to HSV for the pink color #fdc1cc
    pink_rgb = np.uint8([[[253, 193, 204]]])
    pink_hsv = cv2.cvtColor(pink_rgb, cv2.COLOR_RGB2HSV)
    pink_hue = pink_hsv[0][0][0]

    # Define the range of the pink color in HSV
    lower_pink = np.array([pink_hue - 10, 25, 25])
    upper_pink = np.array([pink_hue + 10, 255, 255])

    # Create a mask for the pink color
    mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

    # Define the range of the purple color in HSV
    lower_purple = np.array([140, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # Create a mask for the purple color
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)

    # Define the range of the blue color in HSV
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create a mask for the blue color
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Define the range of the new color (hex #ee0201) in HSV
    lower_Red_color = np.array([0,100, 100])
    upper_Red_color = np.array([10, 255, 255])

    # Create a mask for the new color
    mask_new_color = cv2.inRange(hsv, lower_Red_color, upper_Red_color)

    # Define the range of green color in HSV
    lower_green = np.array([50, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Create a mask for the green color
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Define the range for dark yellow-orange color in HSV
    lower_yellow_orange = np.array([20, 180, 180])  # Adjusted lower bound for orange
    upper_yellow_orange = np.array([30, 255, 255])  # Adjusted upper bound for orange

    # Create a mask for the dark yellow-orange color
    mask_orange = cv2.inRange(hsv, lower_yellow_orange, upper_yellow_orange)

    # Convert hex to RGB
    hex_color = '#bebe49'
    rgb_color = [int(hex_color[i:i+2], 16) for i in (1, 3, 5)]

    # Convert RGB to HSV
    rgb_color = np.uint8([[rgb_color]])
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]

    # Define the range of the target color in HSV
    tolerance = 4
    lower_color = np.array([hsv_color[0] - tolerance, 50, 50])
    upper_color = np.array([hsv_color[0] + tolerance, 255, 255])

    mask_triangle = cv2.inRange(hsv, lower_color, upper_color)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([50, 110, 150])

    # Create a mask for the black color
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    # Combine purple with each specified color
    combined_masks = {
        "IGT_gas": cv2.bitwise_or(mask_pink, mask_purple),
        "Intermediate Pressure (IP)": cv2.bitwise_or(mask_green, mask_purple),
        "Medium Pressure (MP)": cv2.bitwise_or(mask_blue, mask_purple),
        "High Pressure (HP)": cv2.bitwise_or(mask_orange, mask_purple),
        "Low Pressure (LP)": cv2.bitwise_or(mask_new_color, mask_purple),
        "GOLDEN TRIANGLE": cv2.bitwise_or(mask_triangle, mask_purple),
        "Depth of Cover": cv2.bitwise_or(mask_black, mask_purple),
    }

    # Manually define the bounding box coordinates (x_min, y_min, x_max, y_max)
    x_min, y_min, x_max, y_max = 8, 10, 585, 580

    # Apply bounding box mask to each combined mask
    results = {}
    for label, combined_mask in combined_masks.items():
        bbox_mask = np.zeros_like(combined_mask)
        bbox_mask[y_min:y_max, x_min:x_max] = combined_mask[y_min:y_max, x_min:x_max]
        results[label] = bbox_mask

    # Display the masks using Streamlit
    for label, result in results.items():
        st.image(result, caption=label, use_column_width=True)

    # Combine all masks
    combined_mask = np.zeros_like(mask_purple)
    for result in results.values():
        combined_mask = cv2.bitwise_or(combined_mask, result)

    # Apply the bounding box mask to the original image
    result = cv2.bitwise_and(img, img, mask=combined_mask)

    # Draw the bounding box on the original image
    img_with_bbox = img.copy()
    cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Draw the bounding box on the result image
    result_with_bbox = result.copy()
    cv2.rectangle(result_with_bbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the combined mask and the overall result with bounding boxes
    # st.image(combined_mask, caption='Combined Mask', use_column_width=True)
    st.image(cv2.cvtColor(result_with_bbox, cv2.COLOR_BGR2RGB), caption='Overall Result', use_column_width=True)
    # st.image(cv2.cvtColor(img_with_bbox, cv2.COLOR_BGR2RGB), caption='Image with Bounding Boxes', use_column_width=True)
