param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $Args
)
python tools/cap.py $Args
