import grpc
import federated_pb2
import federated_pb2_grpc
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import io
import model


def train_local(model, trainloader, epochs=1):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(epochs):
        for data, target in trainloader:
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model


def run(client_id=1):
    channel = grpc.insecure_channel("localhost:50051")
    stub = federated_pb2_grpc.FederatedServerStub(channel)

    # === Baixar modelo global ===
    response = stub.GetModel(federated_pb2.Empty())
    buffer = io.BytesIO(response.weights)
    state_dict = torch.load(buffer)
    local_model = model.SimpleMLP()
    local_model.load_state_dict(state_dict)
    print(f"[Cliente {client_id}] Modelo global recebido.")

    # === Dataset local (cada cliente pega subconjunto) ===
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)

    # Simular dados diferentes por cliente
    subset = [i for i in range(len(trainset)) if i % 2 == client_id % 2]
    trainloader = torch.utils.data.DataLoader(torch.utils.data.Subset(trainset, subset), batch_size=32, shuffle=True)

    # === Treino local ===
    local_model = train_local(local_model, trainloader, epochs=1)
    print(f"[Cliente {client_id}] Treinamento local finalizado.")

    # === Enviar update ===
    buffer = io.BytesIO()
    torch.save(local_model.state_dict(), buffer)
    ack = stub.SendUpdate(federated_pb2.ModelUpdate(weights=buffer.getvalue()))
    print(f"[Cliente {client_id}] Servidor respondeu: {ack.message}")


if __name__ == "__main__":
    run(client_id=1)