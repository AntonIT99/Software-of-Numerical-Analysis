set key autotitle columnhead
set xrange [0:250]
set yrange [40:100]
plot "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:5 with lines, "C:/Users/alpha/Documents/Qt_Projects/clanu/models/best/echo_ADAM.dat" using 1:6 with lines  
pause -1

