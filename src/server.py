import grpc
import federated_pb2
import federated_pb2_grpc
from concurrent import futures
import torch
import io
import model
import torchvision
import torchvision.transforms as transforms

# Modelo global
global_model = model.SimpleMLP()
client_updates = []
round_number = 0
expected_clients = 2  # quantos sao esperados por rodada

# Dataset de teste (compartilhado pelo servidor para avaliação global)
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

def evaluate_model(model, testloader):
    """Avalia acurácia do modelo global no dataset de teste."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, target in testloader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return 100.0 * correct / total

class FederatedServerServicer(federated_pb2_grpc.FederatedServerServicer):
    def GetModel(self, request, context):
        buffer = io.BytesIO()
        torch.save(global_model.state_dict(), buffer)
        print("[Servidor] Cliente pediu o modelo global.")
        return federated_pb2.ModelResponse(weights=buffer.getvalue())

    def SendUpdate(self, request, context):
        global global_model, client_updates, round_number

        buffer = io.BytesIO(request.weights)
        client_state = torch.load(buffer)
        client_id = context.peer()  # peer = identificação do cliente
        client_updates.append((client_id, client_state))

        print(f"[Servidor] Update recebido de {client_id} ({len(client_updates)}/{expected_clients})")

        # Se todos os clientes mandaram update, executa FedAvg
        if len(client_updates) >= expected_clients:
            round_number += 1
            print(f"\n===== Rodada {round_number} iniciada =====")

            # --- FedAvg ---
            new_state = {}
            for key in client_updates[0][1].keys():
                new_state[key] = sum(update[1][key] for update in client_updates) / len(client_updates)
            global_model.load_state_dict(new_state)

            # Reset updates para próxima rodada
            client_updates = []

            # Avaliar modelo global
            acc = evaluate_model(global_model, testloader)

            # Resumo da rodada
            print("[Servidor] FedAvg concluído")
            print(f"[Servidor] Acurácia global após rodada {round_number}: {acc:.2f}%")
            print(f"===== Rodada {round_number} concluída =====\n")

        return federated_pb2.Ack(message="Update recebido com sucesso!")


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederatedServerServicer_to_server(
        FederatedServerServicer(), server
    )
    server.add_insecure_port("[::]:50051")
    server.start()
    print("Servidor federado rodando em localhost:50051")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
