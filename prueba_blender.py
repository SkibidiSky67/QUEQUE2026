import socket

HOST = "127.0.0.1"
PORT = 5000

print("CONECTANDO...")

try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as cliente:
        cliente.connect((HOST, PORT))

        print("CONECTADO")

        cliente.sendall(b"triste")

        print("MENSAJE ENVIADO: triste")

except Exception as e:
    print("ERROR:", repr(e))