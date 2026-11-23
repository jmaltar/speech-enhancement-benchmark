extern unsigned _estack;
void Reset_Handler(void);
int main(void);
__attribute__((section(".isr_vector"), used))
void * const g_vectors[2] = { (void *)&_estack, (void *)Reset_Handler };
void Reset_Handler(void) { main(); for(;;){} }
/* newlib stubs so --specs=nosys is not needed */
void _exit(int c){(void)c;for(;;){}}
int _close(int f){(void)f;return -1;}
int _fstat(int f, void *s){(void)f;(void)s;return -1;}
int _isatty(int f){(void)f;return 0;}
int _lseek(int f,int o,int w){(void)f;(void)o;(void)w;return -1;}
int _read(int f,char*b,int n){(void)f;(void)b;(void)n;return -1;}
int _write(int f,const char*b,int n){(void)f;(void)b;(void)n;return n;}
int _kill(int p,int s){(void)p;(void)s;return -1;}
int _getpid(void){return 1;}
extern char _ebss;
static char *heap = &_ebss;
void *_sbrk(int incr){ char *p = heap; heap += incr; return p; }
/* Every C++ firmware image carries this; TFLM's file-scope op resolver has a
   non-trivial destructor, so __cxa_atexit references it. */
void *__dso_handle = (void *)&__dso_handle;
