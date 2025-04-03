Requires you setup a developer app on spotify in order to get API credentials

Store those in an .env and update the path at the top of the spotify func library for your save location

This project uses uv as the package manager so you should be able to use it to sync a .venv for your kernel

There are some booleans at the very bottom of the spotify func library to trigger full refreshes of your data

Then point the notebook called analyses at the resulting parquets and you should be good to go
