import urllib.parse
import urllib.request
import os

def generate_qr(url, filename="project_qr.png"):
    """
    Generates a QR code image for a given URL using a free web API.
    No extra libraries like 'qrcode' are needed.
    """
    print(f"Generating QR Code for: {url}")
    
    # Encode the URL for the API
    encoded_url = urllib.parse.quote(url)
    
    # API endpoints (using qrserver.com - reliable and free)
    api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={encoded_url}"
    
    try:
        # Download the image
        with urllib.request.urlopen(api_url) as response:
            with open(filename, 'wb') as out_file:
                out_file.write(response.read())
        
        full_path = os.path.abspath(filename)
        print(f"✅ Success! QR Code saved as: {filename}")
        print(f"📍 Full path: {full_path}")
        
    except Exception as e:
        print(f"❌ Error generating QR Code: {e}")

if __name__ == "__main__":
    link = input("Enter the URL you want to convert to QR Code: ").strip()
    if not link:
        # Default placeholder if no link provided
        link = "https://github.com/your-username/GIS"
        print(f"Using default link: {link}")
        
    generate_qr(link)
