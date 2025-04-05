import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    ConversationHandler,
    filters
)

# Состояния диалога
MENU, PROCESS_IMAGE = range(2)

# Настройки
TOKEN = "7768210415:AAEQh6__p2PIRwo4R7Vwy9BCaudmfm0lGYw"
TEXT_PADDING_TOP = 30
TEMP_DIR = "../temp_files"

# Создаем папку для временных файлов
os.makedirs(TEMP_DIR, exist_ok=True)

# Инициализация моделей
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Модель для поиска кошек
detection_model = models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
detection_model.eval()

# Модель для тепловой карты
heatmap_model = models.mobilenet_v2(pretrained=True).to(device)
heatmap_model.eval()

# Клавиатура меню
menu_keyboard = ReplyKeyboardMarkup(
    [["Найти кошку", "Тепловая карта"]],
    resize_keyboard=True,
    one_time_keyboard=True
)


def get_temp_path(filename):
    """Генерирует путь к временному файлу"""
    return os.path.join(TEMP_DIR, filename)


async def cleanup_temp_files():
    """Очистка временных файлов"""
    for filename in os.listdir(TEMP_DIR):
        file_path = os.path.join(TEMP_DIR, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Ошибка при удалении файла {file_path}: {e}")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cleanup_temp_files()
    await update.message.reply_text(
        "🐱 Привет! Я CatFinderBot!\nВыберите действие:",
        reply_markup=menu_keyboard
    )
    return MENU


async def menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора в меню"""
    choice = update.message.text
    context.user_data["choice"] = choice
    await update.message.reply_text(
        "Отправьте мне фото с кошкой",
        reply_markup=None
    )
    return PROCESS_IMAGE


async def process_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_choice = context.user_data.get("choice", "")
    photo_file = await update.message.photo[-1].get_file()

    # Сохраняем изображение во временную папку
    temp_input_path = get_temp_path("input.jpg")
    await photo_file.download_to_drive(temp_input_path)

    try:
        # Сначала проверяем наличие кошки
        has_cat = await check_for_cat(temp_input_path)

        if not has_cat:
            await update.message.reply_text("Не удалось обнаружить кошку")
            return MENU

        # Только если кошка найдена - выполняем выбранное действие
        if user_choice == "Найти кошку":
            result_path = get_temp_path("result.jpg")
            await find_cat(temp_input_path, result_path)
            await update.message.reply_photo(photo=open(result_path, "rb"))
        elif user_choice == "Тепловая карта":
            result_path = get_temp_path("heatmap.jpg")
            await generate_heatmap(temp_input_path, result_path)
            await update.message.reply_photo(photo=open(result_path, "rb"))

    except Exception as e:
        await update.message.reply_text(f"⚠️ Ошибка: {str(e)}")
    finally:
        # Удаляем временные файлы
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if 'result_path' in locals() and os.path.exists(result_path):
            os.remove(result_path)

    await update.message.reply_text("Выберите следующее действие:", reply_markup=menu_keyboard)
    return MENU


async def check_for_cat(image_path):
    """Проверяет наличие кошки на изображении"""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = detection_model(image_tensor)

    masks = prediction[0]['masks'] > 0.5
    labels = prediction[0]['labels']
    cat_masks = masks[labels == 17]  # 17 - класс "кошка" в COCO

    return len(cat_masks) > 0


async def find_cat(input_path, output_path):
    """Обводит кошку на изображении"""
    image = Image.open(input_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = detection_model(image_tensor)

    masks = prediction[0]['masks'] > 0.7
    labels = prediction[0]['labels']
    cat_masks = masks[labels == 17]

    mask = cat_masks[0].squeeze().cpu().numpy()
    mask = cv2.resize(mask.astype(np.uint8), image.size, interpolation=cv2.INTER_LINEAR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_np = np.array(image)
    result = cv2.drawContours(image_np.copy(), contours, -1, (0, 255, 0), 3)

    Image.fromarray(result).save(output_path)

async def generate_heatmap(input_path, output_path):
    """Генерирует тепловую карту для изображения с исправлениями"""

    # 1. Подготовка изображения
    def preprocess_image(img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0).to(device)

    # 2. Модифицированная версия Grad-CAM
    def get_gradcam(model, image_tensor):
        # Хранилища для градиентов и активаций
        gradients = None
        activations = None

        # Хук для сохранения градиентов
        def backward_hook(module, grad_input, grad_output):
            nonlocal gradients
            gradients = grad_output[0].detach()  # Важно: используем detach()

        # Хук для сохранения активаций
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output.detach()  # Важно: используем detach()

        # Регистрируем хуки на последнем сверточном слое
        target_layer = model.features[-1]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)  # Используем полный хук

        try:
            # Прямой проход
            output = model(image_tensor)
            target_class = output.argmax().item()

            # Обратный проход
            model.zero_grad()
            output[0, target_class].backward()

            # Проверка наличия градиентов и активаций
            if gradients is None or activations is None:
                raise ValueError("Не удалось получить градиенты или активации")

            # Вычисление весовых коэффициентов
            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3], keepdim=True)

            # Взвешивание активаций
            weighted_activations = activations * pooled_gradients

            # Создание тепловой карты
            heatmap = torch.mean(weighted_activations, dim=1, keepdim=True)
            heatmap = F.relu(heatmap)
            heatmap = heatmap / (torch.max(heatmap) + 1e-10)  # Добавляем небольшое значение для избежания деления на 0

            return heatmap.squeeze().cpu().numpy()

        finally:
            # Обязательно удаляем хуки
            forward_handle.remove()
            backward_handle.remove()

    # 3. Наложение тепловой карты на изображение
    def overlay_heatmap(original_img, heatmap):
        # Конвертируем PIL Image в numpy array
        img_np = np.array(original_img)

        # Нормализуем и преобразуем тепловую карту
        heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        heatmap = np.uint8(255 * (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Наложение тепловой карты на оригинальное изображение
        superimposed_img = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)
        return Image.fromarray(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))

    # Основная обработка
    try:
        # Загружаем и обрабатываем изображение
        original_img = Image.open(input_path).convert("RGB")
        image_tensor = preprocess_image(original_img)

        # Получаем тепловую карту
        heatmap = get_gradcam(heatmap_model, image_tensor)

        # Накладываем тепловую карту и сохраняем результат
        result_img = overlay_heatmap(original_img, heatmap)
        result_img.save(output_path)

    except Exception as e:
        print(f"Ошибка при генерации тепловой карты: {str(e)}")
        raise

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await cleanup_temp_files()
    await update.message.reply_text("Действие отменено", reply_markup=menu_keyboard)
    return MENU


def main():
    application = Application.builder().token(TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MENU: [MessageHandler(filters.Regex("^(Найти кошку|Тепловая карта)$"), menu_handler)],
            PROCESS_IMAGE: [MessageHandler(filters.PHOTO, process_image)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    application.add_handler(conv_handler)
    print("Бот запущен...")
    application.run_polling()


if __name__ == "__main__":
    main()