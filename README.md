# CIRA: a CXL memory offloaded runtime.
Can we replace the while loop usleep with couroutine and gain speed up?

```bash
…in 🌐 epyc1 in ~/isca25/CIRA/runtime on  main [✘!?] via C v11.4.0-gcc via △ v3.22.1 
❯ time taskset -c 0 numactl -p 1 ./a.out
handle: 0x762c7c000b90
c=-1520289353

________________________________________________________
Executed in   17.00 secs    fish           external
   usr time    3.71 secs    1.09 millis    3.71 secs
   sys time    1.10 secs    0.14 millis    1.10 secs

…c1 in ~/isca25/CIRA/runtime on  main [✘!?] via C v11.4.0-gcc via △ v3.22.1 took 16s 
❯ g++ -g affinity.cpp
…in 🌐 epyc1 in ~/isca25/CIRA/runtime on  main [✘!?] via C v11.4.0-gcc via △ v3.22.1 
❯ time taskset -c 0 numactl -p 1 ./a.out
handle: 0x775e8c000b90
c=193843754

________________________________________________________
Executed in  141.85 secs    fish           external
   usr time    8.02 secs  706.00 micros    8.02 secs
   sys time   11.42 secs  567.00 micros   11.42 secs
```