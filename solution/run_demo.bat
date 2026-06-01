@echo off
echo ========================================
echo  Smart Traffic Light Optimisation Demo
echo ========================================
echo.
echo Select scenario:
echo   1. AM Rush (Monday 8 AM)
echo   2. PM Rush (Monday 5 PM)
echo   3. Midday (Monday 12 PM)
echo   4. Full Day Cycle
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" python run_demo.py am_rush
if "%choice%"=="2" python run_demo.py pm_rush
if "%choice%"=="3" python run_demo.py midday
if "%choice%"=="4" python run_demo.py full_day

pause
