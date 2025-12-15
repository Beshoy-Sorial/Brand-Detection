import socket

SERVER_IP = "127.0.0.1"
SERVER_PORT = 5555
IMAGE_PATH = "../Cloth/a1.png"

def send_image(image_path):
    # Read image as raw bytes
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    if not img_bytes:
        print("Image file is empty!")
        return

    # Create socket
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((SERVER_IP, SERVER_PORT))

    try:
        # Send image bytes
        client.sendall(img_bytes)

        # Tell the server we are done sending
        client.shutdown(socket.SHUT_WR)

        # Receive response
        response = b""
        while True:
            data = client.recv(4096)
            if not data:
                break
            response += data

        print("Server response:")
        print(response.decode("utf-8"))

    finally:
        client.close()

if __name__ == "__main__":
    send_image(IMAGE_PATH)
