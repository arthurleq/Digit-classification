#include <SFML/Graphics.hpp>
#include <tensorflow/c/c_api.h>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdio>

// --- CONSTANTE ---
const int CANVAS_SIZE   = 280;
const int PANEL_WIDTH   = 200;
const int WINDOW_W      = CANVAS_SIZE + PANEL_WIDTH;
const int WINDOW_H      = CANVAS_SIZE;
const int BRUSH_RADIUS  = 10;
const int IMG_SIZE      = 28;

// --- TensorFlow helpers ---
void free_buffer(void* data, size_t) { free(data); }

TF_Buffer* read_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    size_t size = ftell(f);
    fseek(f, 0, SEEK_SET);
    void* data = malloc(size);
    fread(data, 1, size, f);
    fclose(f);
    TF_Buffer* buf = TF_NewBuffer();
    buf->data       = data;
    buf->length     = size;
    buf->data_deallocator = free_buffer;
    return buf;
}

// --- Resize 280x280 → 28x28 (average pooling) ---
std::vector<float> resize_canvas(const sf::Image& img) {
    std::vector<float> out(IMG_SIZE * IMG_SIZE, 0.f);
    int factor = CANVAS_SIZE / IMG_SIZE;

    float min_val = 1.f, max_val = 0.f;

    for (int row = 0; row < IMG_SIZE; ++row) {
        for (int col = 0; col < IMG_SIZE; ++col) {
            float sum = 0.f;
            for (int dy = 0; dy < factor; ++dy)
                for (int dx = 0; dx < factor; ++dx) {
                    sf::Color c = img.getPixel(col * factor + dx, row * factor + dy);
                    sum += 1.f - (c.r / 255.f);  // 0=white, 1=black
                }
            float val = sum / (factor * factor);
            out[row * IMG_SIZE + col] = val;
        }
    }

    return out;
}

// --- CNN ---
std::vector<float> predict(TF_Session* session, TF_Graph* graph,
                           const std::vector<float>& input_data) {
    std::vector<float> result(10, 0.f);
    TF_Status* status = TF_NewStatus();

    // Resolution of the operations
    TF_Operation* input_oper  = TF_GraphOperationByName(graph, "serving_default_input_layer_1");
    TF_Operation* output_oper = TF_GraphOperationByName(graph, "StatefulPartitionedCall_1");

    if (!input_oper)  { printf("ERROR : input_oper is null !\n");  return result; }
    if (!output_oper) { printf("ERROR : output_oper is null !\n"); return result; }
    printf("Operations found.\n");

    // Input tensor
    int64_t dims[] = {1, IMG_SIZE, IMG_SIZE, 1};
    size_t  nbytes = input_data.size() * sizeof(float);
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_FLOAT, dims, 4, nbytes);
    if (!input_tensor) { printf("ERROR : input_tensor is null !\n"); return result; }
    memcpy(TF_TensorData(input_tensor), input_data.data(), nbytes);
    printf("Input tensor created.\n");

    TF_Output input_op  = {input_oper,  0};
    TF_Output output_op = {output_oper, 0};

    // Verification of the number of outputs of the Identity operation
    int num_outputs = TF_OperationNumOutputs(output_oper);
    printf("Number of outputs of Identity : %d\n", num_outputs);

    TF_Tensor* output_tensor = nullptr;
    printf("Launching TF_SessionRun...\n");
    TF_SessionRun(session, nullptr,
                  &input_op,  &input_tensor,  1,
                  &output_op, &output_tensor, 1,
                  nullptr, 0, nullptr, status);

    printf("SessionRun is over. Status : %s\n", TF_Message(status));

    if (TF_GetCode(status) == TF_OK && output_tensor) {
        float* data = (float*)TF_TensorData(output_tensor);
        for (int i = 0; i < 10; ++i)
            result[i] = data[i];
        TF_DeleteTensor(output_tensor);
        printf("Prediction OK. Top: %d\n", (int)(std::max_element(result.begin(), result.end()) - result.begin()));
    } else {
        printf("ERROR SessionRun : %s\n", TF_Message(status));
    }

    TF_DeleteTensor(input_tensor);
    TF_DeleteStatus(status);
    return result;
}

// --- Main ---
int main() {

    // Loading the SavedModel
    TF_Graph*  graph   = TF_NewGraph();
    TF_Status* status  = TF_NewStatus();
    TF_SessionOptions* opts = TF_NewSessionOptions();

    const char* tags[]  = {"serve"};
    TF_Session* session = TF_LoadSessionFromSavedModel(
        opts, nullptr, "../saved_model", tags, 1, graph, nullptr, status);

    if (TF_GetCode(status) != TF_OK) {
        printf("ERROR loading model : %s\n", TF_Message(status));
        return 1;
    }
    printf("Model loaded successfully.\n");

    // Window SFML
    sf::RenderWindow window(sf::VideoMode(WINDOW_W, WINDOW_H), "Digit Recognizer");
    window.setFramerateLimit(60);
    sf::View view(sf::FloatRect(0, 0, WINDOW_W, WINDOW_H));
    window.setView(view);      

    // Framework of drawing
    sf::RenderTexture canvas;
    canvas.create(CANVAS_SIZE, CANVAS_SIZE);
    canvas.clear(sf::Color::White);
    canvas.display();

    sf::Sprite canvasSprite(canvas.getTexture());

    // Border of the canvas
    sf::RectangleShape border(sf::Vector2f(CANVAS_SIZE, CANVAS_SIZE));
    border.setFillColor(sf::Color::Transparent);
    border.setOutlineColor(sf::Color(80, 80, 80));
    border.setOutlineThickness(2);

    // Left panel (where the information are displayed)
    sf::RectangleShape panel(sf::Vector2f(PANEL_WIDTH, WINDOW_H));
    panel.setPosition(CANVAS_SIZE, 0);
    panel.setFillColor(sf::Color(30, 30, 30));

    // Police
    sf::Font font;
    if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
        printf("ERROR : font not found.\n");
        return 1;
    }

    // Texts in the right panel
    sf::Text titleText("Prediction", font, 16);
    titleText.setFillColor(sf::Color(200, 200, 200));
    titleText.setPosition(CANVAS_SIZE + 10, 10);

    std::vector<sf::Text> predTexts(3, sf::Text("", font, 14));
    for (int i = 0; i < 3; ++i) {
        predTexts[i].setPosition(CANVAS_SIZE + 10, 50 + i * 60);
    }

    sf::Text hintText("Draw a digit", font, 12);
    hintText.setFillColor(sf::Color(120, 120, 120));
    hintText.setPosition(CANVAS_SIZE + 5, WINDOW_H - 30);

    // Initial empty predictions
    std::vector<float> probs(10, 0.f);
    bool hasPrediction = false;

    bool isDrawing = false;

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {

            if (event.type == sf::Event::Closed)
                window.close();

            if (event.type == sf::Event::Resized) {
                sf::View view(sf::FloatRect(0, 0, WINDOW_W, WINDOW_H));
                window.setView(view);
            }

            if (event.type == sf::Event::MouseButtonPressed &&
                event.mouseButton.button == sf::Mouse::Left) {
                isDrawing = true;
                canvas.clear(sf::Color::White);
                canvas.display();
                hasPrediction = false;
            }

            if (event.type == sf::Event::MouseButtonReleased &&
                event.mouseButton.button == sf::Mouse::Left) {
                isDrawing = false;

                // Get the image from the canvas and predict
                sf::Image img = canvas.getTexture().copyToImage();
                auto input    = resize_canvas(img);
                probs         = predict(session, graph, input);
                hasPrediction = true;
            }
        }

        // Drawing
        if (isDrawing) {
            sf::Vector2f pos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

            if (pos.x >= 0 && pos.x < CANVAS_SIZE &&
                pos.y >= 0 && pos.y < CANVAS_SIZE) {
                sf::CircleShape brush(BRUSH_RADIUS);
                brush.setFillColor(sf::Color::Black);
                brush.setOrigin(BRUSH_RADIUS, BRUSH_RADIUS);
                brush.setPosition(pos);
                canvas.draw(brush);
                canvas.display();
            }
        }

        // Updating the prediction panel
        if (hasPrediction) {
            // Top 3 predictions
            std::vector<int> idx(10);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return probs[a] > probs[b]; });

            sf::Color colors[] = {
                sf::Color(255, 220, 50),   // Gold   (1st)
                sf::Color(180, 180, 180),  // Silver (2nd)
                sf::Color(200, 130, 80)    // Bronze (3rd)
            };

            for (int i = 0; i < 3; ++i) {
                int    digit = idx[i];
                float  prob  = probs[digit] * 100.f;
                char   buf[64];
                snprintf(buf, sizeof(buf), "#%d  Chiffre %d\n    %.1f%%", i+1, digit, prob);
                predTexts[i].setString(buf);
                predTexts[i].setFillColor(colors[i]);

                // Progression bars are also updated in the rendering loop below
            }
        }

        // Rendering
        window.clear(sf::Color(50, 50, 50));
        window.draw(canvasSprite);
        window.draw(border);
        window.draw(panel);
        window.draw(titleText);
        window.draw(hintText);

        if (hasPrediction) {
            std::vector<int> idx(10);
            std::iota(idx.begin(), idx.end(), 0);
            std::sort(idx.begin(), idx.end(),
                      [&](int a, int b){ return probs[a] > probs[b]; });

            for (int i = 0; i < 3; ++i) {
                window.draw(predTexts[i]);

                // Progression bars are also updated in the rendering loop below
                float barMaxW = PANEL_WIDTH - 20;
                float barW    = barMaxW * probs[idx[i]];
                sf::RectangleShape bar(sf::Vector2f(barW, 8));
                bar.setPosition(CANVAS_SIZE + 10, 80 + i * 60);
                bar.setFillColor(predTexts[i].getFillColor());
                window.draw(bar);
            }
        }

        window.display();
    }

    // Cleanup TF resources
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteSessionOptions(opts);
    TF_DeleteStatus(status);

    return 0;
}