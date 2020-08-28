set key autotitle columnhead
set xrange [0:250]
set yrange [0:.3]
plot "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:2 with lines, "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:3 with lines  
pause -1

