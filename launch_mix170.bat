@echo off
cd /d C:\Users\chgou\nanogpt_speedrun
echo ==== mix170 launch %DATE% %TIME% ==== >> mix170_train.log
"C:\Users\chgou\AppData\Local\Programs\Python\Python313\python.exe" -u train.py ^
  --data mix --run mix170 --out mix170 --tie --bench --checkpoint ^
  --seq 256 --dim 768 --layers 20 --heads 12 --batch-size 32 ^
  --steps 92000 --eval-every 500 --bench-every 3000 ^
  --ckpt-every 2000 --resume --prompt "The best way to learn" >> mix170_train.log 2>&1
