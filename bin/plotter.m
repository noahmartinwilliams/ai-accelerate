#! /usr/bin/octave -qf

args=argv();
if (length(args) < 2)
	printf("Usage: plotter.m input_file output_file.png\n");
	exit(1);
endif
x=load(args{1}) ;
h=figure(1, 'visible', 'off');
set(h, 'position', [0, 0, 1024, 1024]);
plot(x);
xlabel('mini-batch number', 'fontsize', 32);
ylabel('error value', 'fontsize', 32);
set(gca, 'fontsize', 32);
print(args{2}, '-dpng');
