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

# Function to find specific text and replace with gray color
def replace_text_with_gray(img, page, text_to_replace, gray_color=(128, 128, 128)):
    for block in page.get_text("dict")["blocks"]:
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if text_to_replace.lower() in span["text"].lower():
                        bbox = span["bbox"]
                        x_min, y_min, x_max, y_max = map(int, bbox)
                        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), gray_color, thickness=cv2.FILLED)

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
    image_path = "image05.png"
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
    st.write("\nPDF Details:")
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
        st.write("Risk_warning_found': Found")
        for bbox in caution_boxes:
            x_min, y_min, x_max, y_max = map(int, bbox)
    else:
        st.write("Risk_warning_found': Not Found")

    # Load the image
    img = cv2.imread(image_path)

    # Replace "Overview map of worksite" with gray color
    replace_text_with_gray(img, page, "Overview map of worksite")

    # Draw bounding boxes around white text
    for white_text in white_texts:
        x_min, y_min, x_max, y_max = map(int, white_text["bbox"])
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red bounding box

    # Draw bounding boxes around "RISK OF DEATH OR SERIOUS INJURY" text
    for bbox in caution_boxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blue bounding box

    # Manually define the bounding box coordinates
    x_min, y_min, x_max, y_max = 8, 10, 585, 580

    # Draw the manually defined bounding box on the image
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 0), 2)  # Black bounding box

    # Crop the image to the bounding box
    cropped_img = img[y_min:y_max, x_min:x_max]

    # Convert cropped image to HSV
    hsv_cropped = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)

    # Define color ranges and masks for color detection
    lower_pink = np.array([174, 0, 255])
    upper_pink = np.array([175, 255, 255])
    mask_pink = cv2.inRange(hsv_cropped, lower_pink, upper_pink)

    lower_purple = np.array([140, 50, 50])
    upper_purple = np.array([160, 255, 255])
    mask_purple = cv2.inRange(hsv_cropped, lower_purple, upper_purple)

    lower_red = np.array([0, 150, 150])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv_cropped, lower_red, upper_red)

    lower_green = np.array([40, 150, 150])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv_cropped, lower_green, upper_green)

    lower_orange = np.array([20, 180, 180])
    upper_orange = np.array([20, 255, 255])
    mask_orange = cv2.inRange(hsv_cropped, lower_orange, upper_orange)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv_cropped, lower_blue, upper_blue)

    # Check detection of each color within the bounding box
    detection_status = {
        "IGT_gas detected": np.any(mask_pink),
        "Low Pressure detected": np.any(mask_red),
        "Intermediate Pressure detected": np.any(mask_green),
        "High Pressure detected": np.any(mask_orange),
        "Medium Pressure detected": np.any(mask_blue),
    }

    # Display detection results
    st.write("Detection Results:")
    for color, detected in detection_status.items():
        st.write(f"{color}: {'Yes' if detected else 'No'}")

    # Display the masks
    # st.image(mask_pink, caption='Mask for Pink (Cropped)', use_column_width=True)
    # st.image(mask_red, caption='Mask for Red (Cropped)', use_column_width=True)
    # st.image(mask_green, caption='Mask for Green (Cropped)', use_column_width=True)
    # st.image(mask_orange, caption='Mask for Orange (Cropped)', use_column_width=True)
    # st.image(mask_blue, caption='Mask for Blue (Cropped)', use_column_width=True)

    # Combine all masks
    combined_mask = np.zeros_like(mask_pink)
    combined_mask = cv2.bitwise_or(combined_mask, mask_pink)
    combined_mask = cv2.bitwise_or(combined_mask, mask_red)
    combined_mask = cv2.bitwise_or(combined_mask, mask_green)
    combined_mask = cv2.bitwise_or(combined_mask, mask_orange)
    combined_mask = cv2.bitwise_or(combined_mask, mask_blue)
    combined_mask = cv2.bitwise_or(combined_mask, mask_purple)


    # Apply the combined mask to the cropped image
    result = cv2.bitwise_and(cropped_img, cropped_img, mask=combined_mask)

    # Display the final results for the cropped area
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption='Result', use_column_width=True)

    # Save the result
    output_path = "Result.png"
    cv2.imwrite(output_path, result)
    st.write(f"Saved as: {output_path}")
