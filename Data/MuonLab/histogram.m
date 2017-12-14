load 15/lifetime_raw.txt
index = lifetime_raw(:,1)
time = lifetime_raw(:,2);
error = lifetime_raw(:,3);

hist(time, 500);