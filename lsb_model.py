from PIL import Image
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_score, mean_squared_error as mse_score

class LSBModel:
    def message_to_binary(self, message):
        return ''.join([format(ord(i), '08b') for i in message])

    def encode_image(self, image_path, messages, output_path="encoded_E2.png"):
        image = Image.open(image_path).convert("RGB")
        np_image = np.array(image)

        combined_message = '|'.join(messages) + "#####"
        binary_message = self.message_to_binary(combined_message)

        binary_index = 0
        data_len = len(binary_message)

        for i in range(np_image.shape[0]):
            for j in range(np_image.shape[1]):
                for k in range(3):  
                    if binary_index < data_len:
                        pixel = np_image[i, j, k]
                        pixel_bin = format(pixel, '08b')
                        new_pixel_bin = pixel_bin[:-1] + binary_message[binary_index]
                        np_image[i, j, k] = int(new_pixel_bin, 2)
                        binary_index += 1
            if binary_index >= data_len:
                break

        encoded_image = Image.fromarray(np_image)
        encoded_image.save(output_path)
        return image, encoded_image

    def binary_to_string(self, binary_data):
        chars = [binary_data[i:i + 8] for i in range(0, len(binary_data), 8)]
        decoded = ''.join([chr(int(char, 2)) for char in chars])
        return decoded.split("#####")[0]

    def decode_and_evaluate(self, original_image, encoded_image_path):
        encoded_image = Image.open(encoded_image_path).convert("RGB")
        np_encoded = np.array(encoded_image)

        binary_data = ""
        for i in range(np_encoded.shape[0]):
            for j in range(np_encoded.shape[1]):
                for k in range(3):
                    pixel_bin = format(np_encoded[i, j, k], '08b')
                    binary_data += pixel_bin[-1]

        hidden_message = self.binary_to_string(binary_data)

        original_cv = np.array(original_image)
        encoded_cv = np.array(encoded_image)

        mse = mse_score(original_cv, encoded_cv)
        psnr = psnr_score(original_cv, encoded_cv)
        embedded_bits = len(binary_data)

        height, width, channels = original_cv.shape
        capacity = embedded_bits / (height * width * channels)
        bpp = embedded_bits / (height * width)

        loss_fn = lpips.LPIPS(net='alex')
        o_tensor = torch.tensor(original_cv.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
        e_tensor = torch.tensor(encoded_cv.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
        lpips_val = loss_fn(o_tensor, e_tensor).item()

        nc = np.corrcoef(original_cv.flatten(), encoded_cv.flatten())[0,1]

        return {
            "message": hidden_message,
            "capacity": capacity,
            "mse": mse,
            "psnr": psnr,
            "nc": nc,
            "lpips": lpips_val,
            "bpp": bpp
        }
