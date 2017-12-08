sample = importdata('slinger_modified_2.csv'); % column 2 3 4 zijn xyz acc
sample = sample.data;

index = sample(:,16);
acc_x = sample(:,3);
acc_y = sample(:,4); 
acc_z = sample(:,5);

min = 2400
max = 2800 %9500
%index = index(min:max);
%acc_z = acc_z(min:max);

plot(index,acc_z)
xlabel('t (milliseconds)')
ylabel('X(t)')

% Hoe zit het met de sample frequentie?
Y = fft(acc_z);
f = (0:length(Y)-1)*50/length(Y);
plot(f,abs(Y))
ylim([0 25])% stel limieten in om in te zoomen op frequenties
xlim([-5 55])
grid on

xlabel('Frequentie (Hz)')
ylabel('Amplitude [-]')