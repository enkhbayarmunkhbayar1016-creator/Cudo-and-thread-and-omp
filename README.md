# Cudo-and-thread-and-omp
# ПАРАЛЛЕЛЬ ЭРЭМБЭЛЭЛТИЙН АЛГОРИТМУУДЫН ХАРЬЦУУЛСАН ШИНЖИЛГЭЭ
### Sequential, std::thread, OpenMP болон CUDA хэрэгжүүлэлтүүдийн гүйцэтгэлийн судалгаа

**Огноо:** 2026 оны 04 сар  
**Хичээл:** Параллель программчлал  

---

## 1. ХУРААНГУЙ

Энэхүү бие даалтын ажилд Odd-Even Transposition Sort алгоритмыг дөрвөн өөр технологи ашиглан хэрэгжүүлж, тэдгээрийн гүйцэтгэлийг харьцуулан судалсан. Судалгааны объектууд нь: дараалсан (Sequential) гүйцэтгэл, C++ std::thread (12 thread), OpenMP болон CUDA (GPU) юм. N = 10,000, 100,000, 1,000,000 хэмжээтэй өгөгдөл дээр туршилт хийж, гүйцэтгэлийн цаг, SpeedUp, нийт үйлдлийн тоо, хүрэх гүйцэтгэл зэрэг үзүүлэлтүүдийг гаргав.

Үр дүнгээс харахад N=1,000,000 үед CUDA нь дараалсан алгоритмаас **~215 дахин**, std::thread-ээс **~21 дахин** хурдан ажилласан. Энэ нь GPU-ийн мянга мянган core зэрэг ажиллах боломжоос шалтгаална.

---

## 2. ОНОЛЫН ОЙЛГОЛТУУД

### 2.1 Параллель тооцоолол

Параллель тооцоолол нь нэг даалгаврыг олон процессор эсвэл core дээр зэрэг гүйцэтгэх аргачлал юм. Үндсэн ойлголтууд:

- **Amdahl's Law:** Параллельчлалын хамгийн их SpeedUp-ийг дараах томьёогоор тодорхойлно:
  ```
  SpeedUp = 1 / (S + (1 - S) / P)
  ```
  Энд S = дараалсан хэсгийн харьцаа, P = процессорын тоо.

- **Data Parallelism:** Нэг үйлдлийг өгөгдлийн өөр өөр хэсгүүд дээр зэрэг хэрэгжүүлэх. Odd-Even Sort энэ загварт тохирно.

- **Synchronization Overhead:** Thread буюу core-уудын хооронд координация хийхэд зарцуулах нэмэлт цаг.

### 2.2 Odd-Even Transposition Sort

Odd-Even Transposition Sort нь Bubble Sort-ын параллель хувилбар юм. Алгоритм нь pass бүрт хоёр фазад хуваагдана:

- **Even фаз:** (0,1), (2,3), (4,5), ... хосуудыг зэрэг харьцуулж swap хийнэ
- **Odd фаз:** (1,2), (3,4), (5,6), ... хосуудыг зэрэг харьцуулж swap хийнэ

```
Жишээ: [5, 3, 8, 1, 9, 2]

Even фаз: (5,3)(8,1)(9,2) → [3,5,1,8,2,9]  ← зэрэг
Odd  фаз:  (5,1)(8,2)    → [3,1,5,2,8,9]  ← зэрэг
Even фаз: (3,1)(5,2)(8,9)→ [1,3,2,5,8,9]  ← зэрэг
...
```

**Нарийн төвөгтэй байдал:**
- Нийт pass: N
- Нэг pass-д харьцуулалт: N/2
- Нийт үйлдэл: N × N/2 = **N²/2**
- Цаг: O(N²) — дараалсан, O(N) — P≥N/2 processor-тэй үед

### 2.3 Технологиудын тойм

| Технологи | Санах ой хуваалцах | Core тоо | Хэрэглээ |
|---|---|---|---|
| std::thread | Shared memory | CPU core (12) | Ерөнхий параллель |
| OpenMP | Shared memory | CPU core | Шинжлэх ухааны тооцоолол |
| CUDA | GPU global memory | 1000+ CUDA core | GPU тооцоолол |

---

## 3. АЛГОРИТМЫН ЗОХИОМЖ

### 3.1 Sequential (Дараалсан) хэрэгжүүлэлт

**Pseudo код:**
```
function BubbleSort(arr[0..N-1]):
    for pass = 0 to N-1:
        for i = 0 to N-2:
            if arr[i] > arr[i+1]:
                swap(arr[i], arr[i+1])
```

**Санах ойн түвшин:** Зөвхөн нэг thread, L1 cache дахь элементүүдийг дараалан унших → cache miss их гарна.

### 3.2 std::thread хэрэгжүүлэлт

**Зарчим:** Pass бүрийг T thread-д хуваарилна. Thread бүр өөрийн хэсгийн хосуудыг харьцуулна. Pass-уудын хооронд **Barrier** ашиглан синхрончлол хийнэ.

**Pseudo код:**
```
function ThreadWorker(arr, threadId, numThreads, N, barrier):
    for pass = 0 to N-1:
        phase = pass % 2
        numPairs = (N - phase) / 2
        chunk = numPairs / numThreads
        from = threadId * chunk
        to   = min(from + chunk, numPairs)

        for i = from to to:
            idx = phase + 2 * i
            if arr[idx] > arr[idx+1]:
                swap(arr[idx], arr[idx+1])

        barrier.arrive_and_wait()   ← бүх thread хүлээнэ

function ParallelSort(arr, N):
    barrier = new Barrier(NUM_THREADS)
    threads = []
    for t = 0 to NUM_THREADS-1:
        threads.add(new Thread(ThreadWorker, t, barrier))
    for t in threads: t.join()
```

**Санах ойн ажиллагаа:**
```
┌─────────────────────────────────────┐
│           Shared RAM                │
│  arr[0..N-1]  (бүх thread хуваална)│
└──────┬────────────────────┬─────────┘
       │                    │
  ┌────▼───┐           ┌────▼───┐
  │Thread 0│           │Thread 1│  ...  Thread 11
  │L1 Cache│           │L1 Cache│
  └────────┘           └────────┘
  Pass хооронд Barrier-ээр синхрончлол
```

**Сул тал:** Pass бүрт N удаа Barrier дуудна → синхрончлолын overhead их.

### 3.3 CUDA хэрэгжүүлэлт

**Зарчим:** Pass бүрт N/2 хосыг GPU-ийн олон thread **нэг зэрэг** боловсруулна. Kernel нэг удаа дуудагдахад бүх хосыг зэрэг харьцуулна.

**Pseudo код:**
```
GPU Kernel oddEven(arr, N, phase):
    i   = blockIdx.x * blockDim.x + threadIdx.x  ← thread индекс
    idx = phase + 2 * i
    if idx+1 < N and arr[idx] > arr[idx+1]:
        swap(arr[idx], arr[idx+1])

CPU хост код:
function CudaSort(arr, N):
    dev_arr = GPU_malloc(N * sizeof(int))
    memcpy(CPU → GPU, arr)              ← өгөгдөл дамжуулах

    blockSize = 256
    gridSize  = (N/2 + blockSize - 1) / blockSize

    for pass = 0 to N-1:
        oddEven<<<gridSize, blockSize>>>(dev_arr, N, pass%2)
        ← GPU kernel дуусахыг хүлээхгүй (асинхрон)

    cudaDeviceSynchronize()             ← бүх pass дуусахыг хүлээх
    memcpy(GPU → CPU, arr)              ← үр дүн авах
    GPU_free(dev_arr)
```

**GPU санах ойн бүтэц:**
```
┌──────────────────────────────────────────────┐
│              GPU Global Memory               │
│         dev_arr[0..N-1]  (4MB @ N=1M)       │
└─────┬──────────┬──────────┬──────────────────┘
      │          │          │
 ┌────▼──┐  ┌───▼───┐  ┌───▼───┐
 │SM   0 │  │SM   1 │  │SM   2 │  ...  (Streaming Multiprocessors)
 │256    │  │256    │  │256    │
 │thread │  │thread │  │thread │
 │       │  │       │  │       │
 │Shared │  │Shared │  │Shared │  ← L1/Shared memory (быстр)
 │Memory │  │Memory │  │Memory │
 └───────┘  └───────┘  └───────┘
   Register бүр нэг элемент хадгална
```

**CUDA thread шатлал:**
```
Grid (нэг kernel дуудалт)
 └── Block [0]  Block [1]  ...  Block [gridSize-1]
       └── 256 thread тус бүрд
             └── нэг хос харьцуулна
```

---

## 4. ҮР ДҮН, ХАРЬЦУУЛСАН ШИНЖИЛГЭЭ

### 4.1 Гүйцэтгэлийн цаг (Execution Time)

| N | Sequential (ms) | std::thread (ms) | CUDA (ms) |
|---|---|---|---|
| 10,000 | 103.51 | 182.13 | 115.99 |
| 100,000 | 16,931.53 | 3,104.94 | 324.55 |
| 1,000,000 | ~2,763,625 (тааман) | 273,705.27 | 12,859.21 |

> Sequential N=1M: N=100K цагаас O(N²) хуулиар тооцсон тааман утга

### 4.2 SpeedUp (дараалсан vs параллель)

```
SpeedUp = T_sequential / T_parallel
```

| N | Thread SpeedUp | CUDA SpeedUp |
|---|---|---|
| 10,000 | **0.57×** (удаан!) | **0.89×** |
| 100,000 | **5.45×** | **52.17×** |
| 1,000,000 | **~10.1×** | **~214.9×** |

> N=10,000 үед Thread удаан: синхрончлолын overhead тооцооллоос давна.

### 4.3 Нийт үйлдлийн тоо (Total Operations)

```
Нийт харьцуулалт = N × N/2 = N²/2
```

| N | Нийт харьцуулалт |
|---|---|
| 10,000 | 50,000,000 (50M) |
| 100,000 | 5,000,000,000 (5B) |
| 1,000,000 | 500,000,000,000 (500B) |

### 4.4 CUDA өгөгдөл дамжуулалт (Data Transfer)

```
Дамжуулах өгөгдөл = N × 4 байт × 2 (upload + download)
```

| N | Өгөгдлийн хэмжээ | Чиглэл |
|---|---|---|
| 10,000 | 80 KB | CPU → GPU → CPU |
| 100,000 | 800 KB | CPU → GPU → CPU |
| 1,000,000 | 8 MB | CPU → GPU → CPU |

> Тайлбар: cudaEventRecord нь зөвхөн kernel-ийн цагийг хэмжсэн тул дамжуулалтын цаг тайлангийн цагт ороогүй. Бодит нийт цаг = kernel цаг + transfer цаг.

### 4.5 Хүрэх гүйцэтгэл (Achievable Performance)

```
Performance = Нийт_үйлдэл / Цаг_секунд  [GOPS — Giga Operations Per Second]
```

| Алгоритм | N=100,000 | N=1,000,000 |
|---|---|---|
| Sequential | 0.295 GOPS | ~0.181 GOPS |
| std::thread | 1.612 GOPS | 1.826 GOPS |
| CUDA | **15.40 GOPS** | **38.88 GOPS** |

### 4.6 Харьцуулалтын дүгнэлт хүснэгт

| Үзүүлэлт | Sequential | std::thread | CUDA |
|---|---|---|---|
| Зэрэгцэлийн зэрэг | 1 | 12 thread | ~1000+ core |
| Алгоритм | Bubble Sort | Odd-Even (parallel) | Odd-Even (GPU) |
| N²/2 нийт үйлдэл | Бүгдийг дараалан | 12-аар хуваана | Бүгдийг зэрэг |
| N=1M SpeedUp | 1× (baseline) | ~10.1× | ~215× |
| N=1M Performance | ~0.18 GOPS | 1.83 GOPS | 38.88 GOPS |
| Хамгийн тохиромжтой | Жижиг N | Дунд N | Том N |

---

## 5. ДҮГНЭЛТ

Энэхүү судалгаагаар Odd-Even Transposition Sort алгоритмыг дөрвөн өөр платформ дээр хэрэгжүүлж харьцуулан шинжиллээ. Дараах дүгнэлтэд хүрлээ:

**1. CUDA хамгийн өндөр гүйцэтгэлтэй.**
N=1,000,000 дээр CUDA нь дараалсан алгоритмаас 215 дахин, std::thread-ээс 21 дахин хурдан ажилласан. Энэ нь GPU-ийн мянга мянган core нэг pass-д бүх хосыг зэрэг харьцуулдагтай шууд холбоотой.

**2. std::thread жижиг өгөгдөлд үр дүнгүй.**
N=10,000 үед std::thread нь дараалсан хувилбараас 0.57 дахин буюу **удаан** ажилласан. Шалтгаан нь pass бүрт 10,000 удаа Barrier синхрончлол хийдэгт байна. N том болох тусам энэ overhead харьцангуй багасаж 5-10× SpeedUp хүрнэ.

**3. Алгоритмын сонголт чухал.**
Odd-Even Sort нь параллельчлахад тохиромжтой боловч O(N²) нарийн төвөгтэй байдлаас гарч чадахгүй. Практикт N>1M үед Merge Sort, Radix Sort зэрэг O(N log N) алгоритмыг GPU дээр хэрэгжүүлбэл илүү үр дүнтэй.

**4. N том болох тусам параллельчлалын ашиг нэмэгдэнэ.**
N 10 дахин нэмэгдэхэд Sequential цаг ~163 дахин, CUDA цаг ~40 дахин нэмэгдсэн нь GPU-ийн харьцангуй давуу тал N томрохын хэрээр улам тодорхой болохыг харуулна.

---

## 6. АШИГЛАСАН МАТЕРИАЛЫН ЖАГСААЛТ

1. Kirk, D. B., & Hwu, W. W. (2016). *Programming Massively Parallel Processors: A Hands-on Approach* (3rd ed.). Morgan Kaufmann.

2. Pacheco, P. S. (2011). *An Introduction to Parallel Programming*. Morgan Kaufmann.

3. NVIDIA Corporation. (2024). *CUDA C++ Programming Guide*. https://docs.nvidia.com/cuda/cuda-c-programming-guide/

4. OpenMP Architecture Review Board. (2021). *OpenMP API Specification Version 5.2*. https://www.openmp.org/specifications/

5. Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley.

6. Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *Proceedings of AFIPS Spring Joint Computer Conference*, 483–485.

7. Batcher, K. E. (1968). Sorting networks and their applications. *Proceedings of AFIPS Spring Joint Computer Conference*, 307–314. *(Odd-Even Merge Sort анхны бүтээл)*

8. cppreference.com. (2024). *std::thread — C++ Reference*. https://en.cppreference.com/w/cpp/thread/thread
