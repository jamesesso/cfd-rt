#[cfg(test)]
mod tests {
    use log::info;

    #[test_log::test(pollster::test)]
    async fn test_instance() {
        let instance = wgpu::Instance::new(Default::default());
        let adapter = instance.request_adapter(&Default::default()).await.unwrap();
        let adapter_info = adapter.get_info();
        info!("{}", adapter_info.backend);
    }
}
