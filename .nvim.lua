-- Set colorcolumn to 100 for Python files only
vim.api.nvim_create_autocmd("FileType", {
  pattern = "python",
  callback = function()
    vim.opt_local.colorcolumn = "100"
  end,
})