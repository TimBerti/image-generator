# ControlNet based Image Generator

This project is an API service that generates images based on a given text prompt and an input image. The image generation algorithm is based on the ControlNet model, which uses a combination of text and image inputs to guide the generation process. The API is built using `FastAPI` and can be deployed as a Docker container. To showcase its capabilities, the service also includes a simple `streamlit` UI for interactively generating images.

## Prerequisites
- [Docker](https://docs.docker.com/get-docker/)

## Deployment Guide
1. Download `control_sd15_canny.pth` and move to `model/` directory.
2. Serialize the model:
   ```sh
   python serialize_model.py
   ```
3. Build the image:
   ```sh
   docker build -t image-generator .
   ```
4. Run the container:
   ```sh
   docker run -d -p 8000:8000 -p 8501:8501 --name image-generator image-generator
   ```
   - API: `http://localhost:8000`
   - UI: `http://localhost:8501`

---

# API Documentation

## Endpoint

### POST `/generate`

This endpoint processes an input image along with various generation parameters, runs the image generation algorithm, and returns one or more generated images encoded in base64 format.

## Request Format

The endpoint expects a JSON payload with two main objects: `params` and `image_data`.

### JSON Schema

```json
{
  "params": {
    "prompt": "string",
    "num_samples": "integer",
    "image_resolution": "integer",
    "strength": "float",
    "guess_mode": "boolean",
    "low_threshold": "integer",
    "high_threshold": "integer",
    "ddim_steps": "integer",
    "scale": "float",
    "seed": "integer",
    "eta": "float",
    "a_prompt": "string",
    "n_prompt": "string"
  },
  "image_data": {
    "image_array": "list of integers"
  }
}
```

### Parameters

#### `params` Object

- **prompt** (`string`):  
  The main text prompt to guide image generation.

- **num_samples** (`integer`):  
  Number of images to generate.

- **image_resolution** (`integer`):  
  The target resolution used when processing the input image (typically determines the size to which the image is resized).

- **strength** (`float`):  
  Scaling factor that controls the influence of the input image on the generated output.

- **guess_mode** (`boolean`):  
  When enabled, the system uses a single strength value for all control scales and disables unconditional prompts.

- **low_threshold** (`integer`):  
  Lower threshold for the Canny edge detection algorithm.

- **high_threshold** (`integer`):  
  Upper threshold for the Canny edge detection algorithm.

- **ddim_steps** (`integer`):  
  Number of steps to perform in the DDIM (Denoising Diffusion Implicit Models) sampling algorithm.

- **scale** (`float`):  
  Guidance scale factor that influences how strongly the unconditional prompt is applied during generation.

- **seed** (`integer`):  
  Random seed for reproducibility. If set to -1, a random seed will be generated.

- **eta** (`float`):  
  Hyperparameter of the DDIM algorithm, affecting the noise schedule.

- **a_prompt** (`string`):  
  Additional (positive) attributes to guide the generation process.

- **n_prompt** (`string`):  
  Negative attributes to steer the generation away from undesired qualities.

#### `image_data` Object

- **image_array** (`list`):  
  An array of integers representing the image data. The image is expected to be encoded (e.g., using OpenCV’s `imencode`) and will be decoded internally by the API.

## Response Format

A successful response returns a JSON object containing the generated images. Each image is provided as a base64-encoded PNG string.

### Example Successful Response

```json
{
  "images": [
    "iVBORw0KGgoAAAANSUhEUgAA...", 
    "iVBORw0KGgoAAAANSUhEUgAA..."
  ]
}
```

## Error Handling

- **400 Bad Request**  
  Returned when the provided image data is invalid (e.g., cannot be decoded into a valid image).

  ```json
  {
    "detail": "Invalid image data."
  }
  ```

- **500 Internal Server Error**  
  Returned for any unexpected errors during processing. The error detail will contain a message describing the problem.

  ```json
  {
    "detail": "Error message describing what went wrong"
  }
  ```

## Example Request

Here’s an example JSON request payload to generate images:

```json
{
  "params": {
    "prompt": "A futuristic cityscape at night",
    "num_samples": 2,
    "image_resolution": 256,
    "strength": 1.0,
    "guess_mode": false,
    "low_threshold": 100,
    "high_threshold": 200,
    "ddim_steps": 50,
    "scale": 7.5,
    "seed": 42,
    "eta": 0.0,
    "a_prompt": "vibrant colors, highly detailed",
    "n_prompt": "low quality, blurry"
  },
  "image_data": {
    "image_array": [137, 80, 78, 71, ...]  // Example: PNG encoded image data represented as an array of integers.
  }
}
```
