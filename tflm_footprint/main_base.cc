// The same image WITHOUT TFLM: the subtraction baseline.
extern "C" const unsigned char g_dtln_model[];
extern "C" int main(void) { return (int)g_dtln_model[0]; }
