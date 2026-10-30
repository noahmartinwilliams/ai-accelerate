#! /usr/bin/octave -q

x=load('/tmp/errors.txt') ;
h=figure(1);
set(h, 'position', [0, 0, 1024, 1024]);
plot(x);
xlabel('mini-batch number', 'fontsize', 32);
ylabel('error value', 'fontsize', 32);
set(gca, 'fontsize', 32);
print -dpng 'error_vs_mini_batch_number.png'
