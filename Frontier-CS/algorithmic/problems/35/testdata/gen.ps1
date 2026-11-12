for ($i = 1; $i -le 10; $i++) {
    New-Item -Path "$i.in" -ItemType File -Force | Out-Null
    New-Item -Path "$i.ans" -ItemType File -Force | Out-Null
}
