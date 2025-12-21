import cv2
import sift
from pathlib import Path
import numpy as np
import socket
import threading
import logo_verifer
import struct

descriptores = {}


def load_logos(base_folder: str):
    logos_dict = {}
    base_path = Path(base_folder)

    for brand_folder in base_path.iterdir():
        if brand_folder.is_dir():
            brand_name = brand_folder.name
            logos = [str(file) for file in brand_folder.glob("*") if file.is_file()]
            logos_dict[brand_name] = logos

    return logos_dict


def recv_exact(sock, num_bytes):
    """Receive exactly num_bytes from socket"""
    data = b""
    while len(data) < num_bytes:
        chunk = sock.recv(num_bytes - len(data))
        if not chunk:
            raise ConnectionError("Connection closed before receiving all data")
        data += chunk
    return data


def handle_client(conn, addr):
    """Handle individual client connection in a separate thread"""
    print(f"Connection from {addr}")

    try:
        # Read length prefix (4 bytes, big endian)
        print("Reading length prefix...")
        length_data = recv_exact(conn, 4)
        image_length = struct.unpack('>I', length_data)[0]
        print(f"Expecting image of {image_length} bytes")
        
        # Receive exact amount of image data
        print("Reading image data...")
        img_bytes = recv_exact(conn, image_length)
        print(f"Received {len(img_bytes)} bytes from {addr}")

        # Decode image
        print("Decoding image...")
        np_arr = np.frombuffer(img_bytes, np.uint8)
        cloth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if cloth is None:
            print("Could not decode the image!")
            conn.sendall(b"ERROR: Could not decode image")
            return
        
        print("Processing image...")
        result = logo_verifer.verify_logo(cloth, descriptores)
        
        # Send result
        print(f"Sending result: {result[:100] if len(result) > 100 else result}")
        result_bytes = result.encode("utf-8")
        conn.sendall(result_bytes)
        print(f"Result sent successfully ({len(result_bytes)} bytes)")

    except Exception as e:
        print(f"Error handling client {addr}: {e}")
        import traceback
        traceback.print_exc()
        try:
            conn.sendall(f"ERROR: {str(e)}".encode("utf-8"))
        except:
            pass
    finally:
        print(f"Closing connection with {addr}")
        conn.close()


if __name__ == "__main__":
    folder = "..\\Logos"
    logos_paths = load_logos(folder)
    sift.init()

    descriptores = {}
    for brand, paths in logos_paths.items():
        descriptores[brand] = []
        for path in paths:
            logo = cv2.imread(path)

            if logo is None:
                print(f"Error loading image: {path}")
                continue

            gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
            kp_logo, des_logo = sift.calc_sift(gray_logo)
            descriptores[brand].append((kp_logo, des_logo))

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 5555))
    server.listen(5)

    print(f"Server listening on port 5555...")

    try:
        while True:
            conn, addr = server.accept()
            client_thread = threading.Thread(target=handle_client, args=(conn, addr))
            client_thread.daemon = True
            client_thread.start()

    except KeyboardInterrupt:
        print("\nShutting down server...")
    finally:
        server.close()