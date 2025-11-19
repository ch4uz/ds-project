import os
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

def to_start_case(filename):
    # Remove extension
    name = os.path.splitext(filename)[0]
    # Replace separators with spaces
    name = name.replace('_', ' ').replace('-', ' ')
    # Capitalize first letter of each word
    return name.title()

def create_pdf_from_images(image_dirs, output_pdf):
    print(f"Generating PDF: {output_pdf}")

    with PdfPages(output_pdf) as pdf:
        for image_dir in image_dirs:
            if not os.path.isdir(image_dir):
                print(f"Warning: Directory '{image_dir}' does not exist. Skipping.")
                continue

            # Get list of PNG files
            images = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
            # Sort alphabetically
            images.sort()

            if not images:
                print(f"No PNG images found in {image_dir}")
                continue

            print(f"Found {len(images)} images in {image_dir}...")

            for image_file in images:
                image_path = os.path.join(image_dir, image_file)
                
                # Create a figure
                # Adjust figsize as needed, or make it dynamic based on image aspect ratio
                # For simplicity, we'll use a standard size and let matplotlib handle scaling
                fig, ax = plt.subplots(figsize=(11.69, 8.27)) # A4 landscape roughly
                
                try:
                    img = mpimg.imread(image_path)
                    
                    # Display image
                    ax.imshow(img)
                    
                    # Remove axes
                    ax.axis('off')
                    
                    # Add title
                    title = to_start_case(image_file)
                    ax.set_title(title, fontsize=16, pad=20)
                    
                    # Save the page
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
                    print(f"Added {image_file} as '{title}'")
                    
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    plt.close(fig)

    print(f"PDF generated successfully: {output_pdf}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a PDF from PNG images in one or more directories.")
    parser.add_argument("directories", nargs='+', help="Path(s) to the directory containing PNG images")
    parser.add_argument("--output", "-o", default="output.pdf", help="Path for the output PDF file")
    
    args = parser.parse_args()
    
    create_pdf_from_images(args.directories, args.output)
