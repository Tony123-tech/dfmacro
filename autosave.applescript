-- Auto-Save Script: Presses Cmd+S every 5 seconds
-- Press Ctrl+C in Script Editor or stop the script to cancel

tell application "System Events"
    repeat
        keystroke "s" using command down
        delay 5
    end repeat
end tell
