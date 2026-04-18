Output assets/demo.gif

Set Shell "zsh"
Set FontSize 14
Set Width 900
Set Height 500
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 20ms

Hide
Type "export PROMPT_EOL_MARK='' && source .venv/bin/activate && clear"
Enter
Sleep 2s
Show

Type "phantom -d ./sample_data"
Enter
Sleep 1s

Type "Find the top 5 most Earth-like planets in this dataset (keep your answer short)."
Sleep 500ms
Enter
Sleep 25s

Sleep 5s
