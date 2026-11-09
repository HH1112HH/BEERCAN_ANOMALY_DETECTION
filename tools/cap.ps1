param(
    [Parameter(ValueFromRemainingArguments = $true)]
    $Args
)
python cap.py $Args
