sample = importdata('call_1.csv'); % column 2 3 4 zijn xyz acc

time = sample(:,1);
acc_x = sample(:,3);
acc_y = sample(:,4); 
acc_z = sample(:,5);

z = sin(acc_x)+cos(acc_y);

surf(acc_x, acc_y, z)

xlabel('Frequentie (Hz)')
ylabel('Amplitude [-]')
grid on

