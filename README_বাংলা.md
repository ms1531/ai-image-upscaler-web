# AI Image Upscaler — Universal CPU Portable

এই ভার্সনটি অফিস/অন্য Windows PC-তে চালানোর জন্য compatibility-first করে বানানো।

## কী পরিবর্তন করা হয়েছে
- আলাদা Python install লাগবে না — portable Python ফোল্ডারের ভিতরেই আছে।
- GPU / DirectML force করা হয়নি; AI inference CPU-only করা হয়েছে, যাতে আলাদা GPU driver/CUDA/DirectML dependency না লাগে।
- Runtime DLL খুঁজে পাওয়ার জন্য launcher PATH ঠিক করে দেয়।
- `MSVCP140.dll` app-local fallback রাখা হয়েছে।
- চালুর আগে ONNX Runtime test হয়; সমস্যা হলে `runtime_test.log` তৈরি হবে।

## ব্যবহার
1. ZIP পুরোটা Extract করুন। OneDrive/ZIP-এর ভিতর থেকে সরাসরি চালাবেন না।
2. `START.bat` ডাবল-ক্লিক করুন।
3. চাইলে `OPEN_APP.vbs` দিয়েও খুলতে পারেন।
4. সমস্যা হলে `CHECK_PC.bat` চালিয়ে স্ক্রিনশট দিন।

## Compatibility
- 64-bit Windows 10/11 লক্ষ্য করে তৈরি।
- আলাদা Python install দরকার নেই।
- CPU mode হওয়ায় GPU mode-এর তুলনায় ধীর হতে পারে, কিন্তু compatibility বেশি।

> নোট: পৃথিবীর প্রতিটি Windows PC-তে 100% guarantee করা যায় না (যেমন 32-bit Windows, খুব পুরোনো OS/CPU, damaged Windows runtime), তবে আগের DirectML/GPU-dependent build-এর তুলনায় এই build অনেক বেশি portable।
