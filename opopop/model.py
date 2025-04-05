import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os
import time


def train_model():
    # 1. Проверка доступности GPU
    print("\n" + "=" * 50)
    print("ИНИЦИАЛИЗАЦИЯ ОБУЧЕНИЯ")
    print("=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nИспользуемое устройство: {device}")
    if device.type == 'cuda':
        print(f"Название GPU: {torch.cuda.get_device_name(0)}")
        print(f"Доступно памяти: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB")

    # 2. Пути к данным
    print("\n" + "-" * 50)
    print("ЗАГРУЗКА ДАННЫХ")
    print("-" * 50)
    dataset_path = "../animals-10"
    train_path = os.path.join(dataset_path, "train")
    val_path = os.path.join(dataset_path, "val")

    # 3. Проверка структуры данных
    print("\nПроверка структуры данных:")
    print(f"Train директория: {train_path}")
    print(f"Val директория: {val_path}")
    print("\nКлассы в train:", os.listdir(train_path))
    print("Классы в val:", os.listdir(val_path))

    # 4. Аугментации и трансформы
    print("\n" + "-" * 50)
    print("ПОДГОТОВКА ДАННЫХ")
    print("-" * 50)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    print("\nТрансформации для train:")
    print(train_transform)
    print("\nТрансформации для val:")
    print(val_transform)

    # 5. Загрузка данных
    print("\nЗагрузка датасетов...")
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)
    print(f"\nРазмер train датасета: {len(train_dataset)} изображений")
    print(f"Размер val датасета: {len(val_dataset)} изображений")
    print(f"Количество классов: {len(train_dataset.classes)}")
    print(f"Метки классов: {train_dataset.classes}")

    # 6. DataLoader
    print("\nСоздание DataLoader...")
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"\nРазмер батча: {batch_size}")
    print(f"Количество батчей в train: {len(train_loader)}")
    print(f"Количество батчей в val: {len(val_loader)}")

    # 7. Модель (EfficientNet)
    print("\n" + "-" * 50)
    print("ПОДГОТОВКА МОДЕЛИ")
    print("-" * 50)
    print("\nЗагрузка предобученной модели EfficientNet-B3...")
    model = models.efficientnet_b3(pretrained=True)

    # Замораживаем слои
    print("\nЗамораживаем все слои...")
    for param in model.parameters():
        param.requires_grad = False

    # Размораживаем последние слои
    print("Размораживаем последние 5 слоев...")
    for param in model.features[-5:].parameters():
        param.requires_grad = True

    # Заменяем классификатор
    num_classes = len(train_dataset.classes)
    print(f"\nЗаменяем классификатор для {num_classes} классов...")
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, num_classes)
    )

    model = model.to(device)
    print("\nАрхитектура модели:")
    print(model)

    # 8. Функции для обучения
    def train(model, loader, criterion, optimizer, epoch):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Вывод прогресса каждые 10 батчей
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader):
                print(f"Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(loader)} | "
                      f"Loss: {loss.item():.4f} | Acc: {100 * correct / total:.2f}% | "
                      f"Time: {time.time() - start_time:.2f}s")

        return running_loss / len(loader), correct / total

    def validate(model, loader, criterion):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return running_loss / len(loader), correct / total, all_preds, all_labels

    # 9. Обучение модели
    print("\n" + "=" * 50)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 50)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW([
        {'params': model.features[-5:].parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 3e-4}
    ], weight_decay=0.01)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    best_val_loss = float('inf')
    patience = 3
    no_improve = 0

    epochs = 15
    train_history = {'loss': [], 'accuracy': []}
    val_history = {'loss': [], 'accuracy': []}

    print("\nПараметры обучения:")
    print(f"Количество эпох: {epochs}")
    print(f"Размер батча: {batch_size}")
    print(f"Оптимизатор: {optimizer.__class__.__name__}")
    print(f"Функция потерь: {criterion.__class__.__name__}")
    print("\nНачинаем обучение...")

    for epoch in range(epochs):
        epoch_start = time.time()
        print("\n" + "-" * 30)
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 30)

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion)

        # Логирование
        train_history['loss'].append(train_loss)
        train_history['accuracy'].append(train_acc)
        val_history['loss'].append(val_loss)
        val_history['accuracy'].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"\nИтоги эпохи {epoch + 1}:")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
        print(f"Время эпохи: {epoch_time:.2f} секунд")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save(model.state_dict(), '../best_model.pth')
            print(f"Val loss улучшился! Сохраняем модель как 'best_model.pth'")
        else:
            no_improve += 1
            print(f"Val loss не улучшился ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"Ранняя остановка после {epoch + 1} эпох!")
                break

        scheduler.step(val_loss)
        print(f"Текущий LR: {optimizer.param_groups[0]['lr']:.2e}")

    # 10. Визуализация результатов
    print("\n" + "=" * 50)
    print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_history['loss'], label='Train Loss')
    plt.plot(val_history['loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_history['accuracy'], label='Train Acc')
    plt.plot(val_history['accuracy'], label='Val Acc')
    plt.title('Accuracy')
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=train_dataset.classes,
                yticklabels=train_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # 11. Сохранение модели
    print("\n" + "=" * 50)
    print("СОХРАНЕНИЕ МОДЕЛИ")
    print("=" * 50)

    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': train_dataset.class_to_idx,
        'classes': train_dataset.classes,
        'transform': val_transform
    }, 'animals_classifier.pth')

    print("\nОбучение завершено!")
    print(f"Лучшая модель сохранена как: animals_classifier.pth")
    print(f"Используемые классы: {train_dataset.classes}")
    print(f"Лучшая Val Loss: {best_val_loss:.4f}")
    print(f"Лучшая Val Accuracy: {max(val_history['accuracy']):.2%}")

if __name__ == '__main__':
    train_model()