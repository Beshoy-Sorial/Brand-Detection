import cv2
from pathlib import Path
import numpy as np
import socket
import threading
import pickle

sift = cv2.SIFT_create(
    nOctaveLayers=5,
    sigma=1.414
)

descriptores = {}


CACHE_FILE = "logo_descriptors.pkl"

def save_descriptors_cache(descriptores):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(descriptores, f)
    print(f"Saved descriptors cache to {CACHE_FILE}")

def load_descriptors_cache():
    if Path(CACHE_FILE).exists():
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    return None

def load_logos(base_folder: str):
    logos_dict = {}
    base_path = Path(base_folder)

    for brand_folder in base_path.iterdir():
        if brand_folder.is_dir():
            brand_name = brand_folder.name
            logos = [str(file) for file in brand_folder.glob("*") if file.is_file()]
            logos_dict[brand_name] = logos

    return logos_dict

def handle_client(conn, addr):
    """Handle individual client connection in a separate thread"""
    print(f"Connection from {addr}")
    
    img_bytes = b""  
    
    try:
        while True:
            data = conn.recv(4096)
            if not data:
                break
            img_bytes += data

        print(f"Received {len(img_bytes)} bytes from {addr}")
        
        if not img_bytes:
            print(f"No data received from {addr}")
            conn.sendall(b"ERROR: No data received")
            return

        # Decode image
        print("Decoding image...")
        np_arr = np.frombuffer(img_bytes, np.uint8)   
        cloth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  

        if cloth is None:
            print("Could not decode the image!")
            conn.sendall(b"ERROR: Could not decode image")
            return
        
        result = photo_matcher(cloth)
        send_result(conn, result)
        
    except Exception as e:
        print(f"Error handling client {addr}: {e}")
        try:
            conn.sendall(f"ERROR: {str(e)}".encode('utf-8'))
        except:
            pass
    finally:
        conn.close()
        print(f"Connection closed with {addr}")

def send_result(conn, result_message):
    conn.sendall(result_message.encode('utf-8'))

def photo_matcher(cloth):
    return "UNKNOWN BRAND"

if __name__ == "__main__":
    folder = "Logos"  
    logos_paths = load_logos(folder)
    

    descriptores = load_descriptors_cache()

    # Load all logo descriptors
    if descriptores is None:
        for brand, paths in logos_paths.items():
            descriptores[brand] = []
            for path in paths:
                logo = cv2.imread(path)
                
                if logo is None:
                    print(f"Error loading image: {path}")
                    continue

                gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
                kp_logo, des_logo = sift.detectAndCompute(logo, None)
                descriptores[brand].append((kp_logo, des_logo))
                
        save_descriptors_cache(descriptores)
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  
    server.bind(("127.0.0.1", 5555))
    server.listen(5)  
    
    print(f"Server listening on port 5555...")
    
    try:
        while True:
            # Accept new connection
            conn, addr = server.accept()
            
            # Spawn a new thread to handle this client
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.daemon = True  # Thread will close when main program exits
            client_thread.start()
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.close()